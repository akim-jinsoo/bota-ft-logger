"""Read Bota Systems Serial sensor and log to CSV

See the following GitLab page for the drivers (https://gitlab.com/botasys/legacy/python_interface)
Follow the readme and then add this file to the examples folder.

- Defaults to COM9 if no arguments are provided (great for running in Spyder).
- Writes a timestamped CSV with a header in the current folder unless --outfile is given.
- Producer/consumer threading: serial parsing is real-time, CSV writes are buffered.
- Clean shutdown on Ctrl+C.

Usage (terminal):
    cd "Documents\Python\BOTA Systems\python_interface-main\examples"
    python bota_serial_example_log.py COM9 --baud 460800 --outfile ".\Grasp_Force_Testing_Encoder_Tuning\T2.csv"
"""

import sys
import struct
import time
import threading
import csv
import datetime as dt
import queue
import argparse
import os
import platform
from collections import namedtuple

import serial
from crc import Calculator, Configuration


class BotaSerialSensor:
    BOTA_PRODUCT_CODE = 123456
    DEFAULT_BAUDERATE = 460800
    SINC_LENGTH = 256
    CHOP_ENABLE = 0
    FAST_ENABLE = 0
    FIR_DISABLE = 1
    TEMP_COMPENSATION = 0   # 0: Disabled (recommended), 1: Enabled
    USE_CALIBRATION = 1     # 1: calibration matrix active, 0: raw measurements
    DATA_FORMAT = 0         # 0: binary, 1: CSV  (we keep binary and unpack)
    BAUDERATE_CONFIG = 4    # 0: 9600, 1: 57600, 2: 115200, 3: 230400, 4: 460800
    FRAME_HEADER = b'\xAA'
    time_step = 0.005       # updated after config

    def __init__(self, port, baud=None, outfile_path=None):
        self._port = self._normalize_port_name(port)
        self._baud = baud if baud is not None else self.DEFAULT_BAUDERATE

        self._ser = serial.Serial()
        self._pd_thread_stop_event = threading.Event()
        self._writer_stop_event = threading.Event()
        self._queue = queue.Queue(maxsize=2000)  # buffer frames for CSV writer

        DeviceSet = namedtuple('DeviceSet', 'name product_code config_func')
        self._expected_device_layout = {
            0: DeviceSet('BFT-SENS-SER-M8', self.BOTA_PRODUCT_CODE, self.bota_sensor_setup)
        }

        # latest values (also mirrored to CSV)
        self._status = None
        self._fx = 0.0
        self._fy = 0.0
        self._fz = 0.0
        self._mx = 0.0
        self._my = 0.0
        self._mz = 0.0
        self._timestamp = 0.0
        self._temperature = 0.0

        # CSV setup
        if outfile_path is None:
            stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            outfile_path = f"bota_readings_{stamp}.csv"
        self._outfile_path = outfile_path
        self._csv_file = None
        self._csv_writer = None

    @staticmethod
    def _normalize_port_name(port: str) -> str:
        """On Windows, ensure COM10+ works; pyserial usually handles 'COM10', but the
        extended form '\\\\.\\COM10' is safest with some stacks."""
        if platform.system().lower().startswith("win"):
            name = port.upper().strip()
            # Accept both 'COM9' and '\\.\COM9'; only convert if missing prefix and >=10
            if name.startswith(r"\\.\COM"):
                return name
            if name.startswith("COM"):
                try:
                    n = int(name[3:])
                    if n >= 10:
                        return r"\\.\{}".format(name)
                except ValueError:
                    pass
        return port

    def bota_sensor_setup(self):
        print("Trying to setup the sensor.")
        # Wait for streaming of data
        out = self._ser.read_until(b'App Init')
        if not self.contains_bytes(b'App Init', out):
            print("Sensor not streaming, check if correct port selected!")
            return False
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        # Go to CONFIG mode
        self._ser.write(b'C')
        out = self._ser.read_until(b'r,0,C,0')
        if not self.contains_bytes(b'r,0,C,0', out):
            print("Failed to go to CONFIG mode.")
            return False

        # Communication setup
        comm_setup = f"c,{self.TEMP_COMPENSATION},{self.USE_CALIBRATION},{self.DATA_FORMAT},{self.BAUDERATE_CONFIG}"
        self._ser.write(comm_setup.encode('ascii'))
        out = self._ser.read_until(b'r,0,c,0')
        if not self.contains_bytes(b'r,0,c,0', out):
            print("Failed to set communication setup.")
            return False
        self.time_step = 0.00001953125 * self.SINC_LENGTH
        print(f"Timestep: {self.time_step}")

        # Filter setup
        filter_setup = f"f,{self.SINC_LENGTH},{self.CHOP_ENABLE},{self.FAST_ENABLE},{self.FIR_DISABLE}"
        self._ser.write(filter_setup.encode('ascii'))
        out = self._ser.read_until(b'r,0,f,0')
        if not self.contains_bytes(b'r,0,f,0', out):
            print("Failed to set filter setup.")
            return False

        # Go to RUN mode
        self._ser.write(b'R')
        out = self._ser.read_until(b'r,0,R,0')
        if not self.contains_bytes(b'r,0,R,0', out):
            print("Failed to go to RUN mode.")
            return False

        return True

    @staticmethod
    def contains_bytes(subsequence, sequence):
        return subsequence in sequence

    def _processdata_thread(self):
        """Producer: reads binary frames from serial and pushes rows to queue."""
        crc16X25Configuration = Configuration(16, 0x1021, 0xFFFF, 0xFFFF, True, True)
        crc_calculator = Calculator(crc16X25Configuration)

        while not self._pd_thread_stop_event.is_set():
            frame_synced = False

            # Try to sync to a valid frame
            while not frame_synced and not self._pd_thread_stop_event.is_set():
                possible_header = self._ser.read(1)
                if self.FRAME_HEADER == possible_header:
                    data_frame = self._ser.read(34)
                    crc16_ccitt_frame = self._ser.read(2)
                    if len(data_frame) != 34 or len(crc16_ccitt_frame) != 2:
                        continue
                    crc16_ccitt = struct.unpack_from('<H', crc16_ccitt_frame, 0)[0]
                    checksum = crc_calculator.checksum(data_frame)
                    if checksum == crc16_ccitt:
                        frame_synced = True
                    else:
                        # shift by one byte and keep searching
                        self._ser.read(1)

            # Stream frames
            while frame_synced and not self._pd_thread_stop_event.is_set():
                frame_header = self._ser.read(1)
                if frame_header != self.FRAME_HEADER:
                    print("Lost sync")
                    frame_synced = False
                    break

                data_frame = self._ser.read(34)
                crc16_ccitt_frame = self._ser.read(2)
                if len(data_frame) != 34 or len(crc16_ccitt_frame) != 2:
                    print("Incomplete frame")
                    frame_synced = False
                    break

                crc16_ccitt = struct.unpack_from('<H', crc16_ccitt_frame, 0)[0]
                checksum = crc_calculator.checksum(data_frame)
                if checksum != crc16_ccitt:
                    print("CRC mismatch received")
                    frame_synced = False
                    break

                # Unpack payload (little-endian)
                self._status = struct.unpack_from('<H', data_frame, 0)[0]
                self._fx = struct.unpack_from('<f', data_frame, 2)[0]
                self._fy = struct.unpack_from('<f', data_frame, 6)[0]
                self._fz = struct.unpack_from('<f', data_frame, 10)[0]
                self._mx = struct.unpack_from('<f', data_frame, 14)[0]
                self._my = struct.unpack_from('<f', data_frame, 18)[0]
                self._mz = struct.unpack_from('<f', data_frame, 22)[0]
                self._timestamp = struct.unpack_from('<I', data_frame, 26)[0]
                self._temperature = struct.unpack_from('<f', data_frame, 30)[0]

                # Create a CSV row with a host-time for traceability
                host_time = time.time()
                row = [
                    host_time,          # epoch seconds (float)
                    self._timestamp,    # device timestamp (uint32)
                    self._status,
                    self._fx, self._fy, self._fz,
                    self._mx, self._my, self._mz,
                    self._temperature
                ]

                # Non-blocking if queue is full: drop oldest to keep up
                try:
                    self._queue.put(row, timeout=0.01)
                except queue.Full:
                    try:
                        _ = self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(row)
                    except queue.Full:
                        pass

    def _writer_thread(self):
        """Consumer: writes queued rows to CSV."""
        # Open file with newline='' to avoid blank lines on Windows
        self._csv_file = open(self._outfile_path, mode='w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        # Header
        self._csv_writer.writerow([
            "host_time_s", "device_timestamp", "status",
            "Fx", "Fy", "Fz", "Mx", "My", "Mz", "temperature_C"
        ])
        self._csv_file.flush()

        try:
            while not self._writer_stop_event.is_set() or not self._queue.empty():
                try:
                    row = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._csv_writer.writerow(row)
                # Flush every row for safety; for higher throughput you can buffer and flush less often
                self._csv_file.flush()
        finally:
            if self._csv_file:
                self._csv_file.flush()
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    def _my_loop(self):
        """Console printout at ~4 Hz for quick monitoring."""
        try:
            while True:
                print('Run my loop')
                print(f"Status {self._status}")
                print(f"Fx {self._fx}")
                print(f"Fy {self._fy}")
                print(f"Fz {self._fz}")
                print(f"Mx {self._mx}")
                print(f"My {self._my}")
                print(f"Mz {self._mz}")
                print(f"Timestamp {self._timestamp}")
                print(f"Temperature {self._temperature}\n")
                time.sleep(0.25)
        except KeyboardInterrupt:
            print('stopped')

    def run(self):
        self._ser.baudrate = self._baud
        self._ser.port = self._port
        self._ser.timeout = 10

        try:
            self._ser.open()
            print(f"Opened serial port {self._port} @ {self._baud} baud")
        except Exception as e:
            raise BotaSerialSensorError(f'Could not open port {self._port}: {e!s}')

        if not self._ser.is_open:
            raise BotaSerialSensorError('Could not open port')

        if not self.bota_sensor_setup():
            print('Could not setup sensor!')
            self._ser.close()
            return

        # Start threads
        proc_thread = threading.Thread(target=self._processdata_thread, name="bota_reader", daemon=True)
        writer_thread = threading.Thread(target=self._writer_thread, name="bota_csv_writer", daemon=True)
        proc_thread.start()
        writer_thread.start()

        print(f"Logging to CSV: {os.path.abspath(self._outfile_path)}")

        try:
            self._my_loop()
        finally:
            # Stop threads and clean up
            self._pd_thread_stop_event.set()
            self._writer_stop_event.set()
            proc_thread.join(timeout=2.0)
            writer_thread.join(timeout=2.0)
            if self._ser and self._ser.is_open:
                self._ser.close()

    @staticmethod
    def _sleep(duration, get_now=time.perf_counter):
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()


class BotaSerialSensorError(Exception):
    def __init__(self, message):
        super(BotaSerialSensorError, self).__init__(message)
        self.message = message


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Read Bota Serial sensor and log to CSV.")
    parser.add_argument("port", nargs="?", default="COM9",
                        help='Serial port (e.g., COM9 or \\\\.\\COM10 on Windows). Defaults to COM9.')
    parser.add_argument("--outfile", help="CSV output file path (default: timestamped in CWD)", default=None)
    parser.add_argument("--baud", type=int, default=BotaSerialSensor.DEFAULT_BAUDERATE,
                        help=f"Serial baud rate (default: {BotaSerialSensor.DEFAULT_BAUDERATE})")
    return parser.parse_args(argv)


if __name__ == '__main__':
    print('bota_serial_example started')

    try:
        # Works nicely in Spyder: with no args, defaults to COM9
        args = _parse_args(sys.argv[1:])
        sensor = BotaSerialSensor(args.port, baud=args.baud, outfile_path=args.outfile)
        sensor.run()
    except BotaSerialSensorError as expt:
        print('bota_serial_example failed: ' + expt.message)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(0)