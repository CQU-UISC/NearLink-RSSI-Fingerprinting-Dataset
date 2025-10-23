import serial
import time

# Serial port configuration
serial_port = '/dev/ttyUSB1'
baud_rate = 115200  # Default baud rate, adjust if needed
timeout = 1  # Read timeout in seconds

# Open serial port
try:
    ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
except serial.SerialException:
    exit(1)

# Prepare output file
output_file = '4_75-2_25.txt'
start_time = time.time()
duration = 120  # Duration in seconds (2 minutes)

try:
    with open(output_file, 'w') as f:
        while (time.time() - start_time) < duration:
            # Read serial data
            if ser.inWaiting() > 0:
                data = ser.readline().decode('utf-8', 'ignore').strip()
                if data:
                    # Write to file
                    f.write(data + '\n')
            time.sleep(0.01)  # Short sleep to reduce CPU usage
except KeyboardInterrupt:
    pass
except Exception:
    pass
finally:
    # Close serial port
    ser.close()

