import serial
import time
import subprocess
import os

def read_serial_and_save(port, filename, duration=2):
    try:
        print "Attempting to open serial port %s" % port
        ser = serial.Serial(port, baudrate=115200, timeout=1)
        time.sleep(1)
        print "Serial port %s opened successfully" % port
        start_time = time.time()
        data = []
        while (time.time() - start_time) < duration:
            if ser.in_waiting > 0:
                try:
                    raw_data = ser.readline()
                    line = raw_data.decode('utf-8', errors='ignore').strip()
                    if line:
                        data.append(line)
                except UnicodeDecodeError, e:
                    print "Decode error in serial data: %s" % str(e)
                    data.append(raw_data.encode('hex'))
        ser.close()
        if not data:
            print "No data received from %s" % port
        with open(filename, 'w') as f:
            for line in data:
                f.write(line + '\n')
        print "Saved %d lines to %s" % (len(data), filename)
    except Exception, e:
        print "Error reading serial port %s: %s" % (port, str(e))

def execute_script(script_name):
    script_path = os.path.join('/home/agilex/catkin_ws/src/scripts/fingerprint_scenario1', script_name)
    try:
        print "Executing script %s" % script_name
        retcode = subprocess.call(['python', script_path])
        if retcode != 0:
            print "Error executing %s: return code %d" % (script_name, retcode)
        else:
            print "Script %s executed successfully" % script_name
    except Exception, e:
        print "Error executing %s: %s" % (script_name, str(e))

def main():
    base_path = '/home/agilex/catkin_ws/src/scripts/fingerprint_scenario1'
    serial_port = '/dev/ttyUSB0'

    read_serial_and_save(serial_port, os.path.join(base_path, '50_50.txt'))

    for i in range(8):
        execute_script('scout_forward_50cm.py')
        filename = os.path.join(base_path, '50_%d.txt' % (100 + i*50))
        read_serial_and_save(serial_port, filename)

    execute_script('scout_turnright_90.py')
    execute_script('scout_turnforward_50cm.py')
    execute_script('scout_turnleft_90.py')
    execute_script('scout_forward_3cm.py')
    read_serial_and_save(serial_port, os.path.join(base_path, '100_450.txt'))

    for i in range(8):
        execute_script('scout_backward_50cm.py')
        filename = os.path.join(base_path, '100_%d.txt' % (400 - i*50))
        read_serial_and_save(serial_port, filename)

    execute_script('scout_turnright_90.py')
    execute_script('scout_turnforward_50cm.py')
    execute_script('scout_turnleft_90.py')
    execute_script('scout_forward_3cm.py')
    read_serial_and_save(serial_port, os.path.join(base_path, '150_50.txt'))

    for i in range(8):
        execute_script('scout_forward_50cm.py')
        filename = os.path.join(base_path, '150_%d.txt' % (100 + i*50))
        read_serial_and_save(serial_port, filename)

    execute_script('scout_turnright_90.py')
    execute_script('scout_turnforward_50cm.py')
    execute_script('scout_turnleft_90.py')
    execute_script('scout_forward_3cm.py')
    read_serial_and_save(serial_port, os.path.join(base_path, '200_450.txt'))

    for i in range(8):
        execute_script('scout_backward_50cm.py')
        filename = os.path.join(base_path, '200_%d.txt' % (400 - i*50))
        read_serial_and_save(serial_port, filename)

    execute_script('scout_turnright_90.py')
    execute_script('scout_turnforward_50cm.py')
    execute_script('scout_turnleft_90.py')
    execute_script('scout_forward_3cm.py')
    read_serial_and_save(serial_port, os.path.join(base_path, '250_50.txt'))

    for i in range(8):
        execute_script('scout_forward_50cm.py')
        filename = os.path.join(base_path, '250_%d.txt' % (100 + i*50))
        read_serial_and_save(serial_port, filename)

    execute_script('scout_turnright_90.py')
    execute_script('scout_turnforward_50cm.py')
    execute_script('scout_turnleft_90.py')
    execute_script('scout_forward_3cm.py')
    read_serial_and_save(serial_port, os.path.join(base_path, '300_450.txt'))

    for i in range(8):
        execute_script('scout_backward_50cm.py')
        filename = os.path.join(base_path, '300_%d.txt' % (400 - i*50))
        read_serial_and_save(serial_port, filename)

    execute_script('scout_turnright_90.py')
    execute_script('scout_turnforward_50cm.py')
    execute_script('scout_turnleft_90.py')
    execute_script('scout_forward_3cm.py')
    read_serial_and_save(serial_port, os.path.join(base_path, '350_50.txt'))

    for i in range(8):
        execute_script('scout_forward_50cm.py')
        filename = os.path.join(base_path, '350_%d.txt' % (100 + i*50))
        read_serial_and_save(serial_port, filename)

    execute_script('scout_turnright_90.py')
    execute_script('scout_turnforward_50cm.py')
    execute_script('scout_turnleft_90.py')
    execute_script('scout_forward_3cm.py')
    read_serial_and_save(serial_port, os.path.join(base_path, '400_450.txt'))

    for i in range(8):
        execute_script('scout_backward_50cm.py')
        filename = os.path.join(base_path, '400_%d.txt' % (400 - i*50))
        read_serial_and_save(serial_port, filename)

    execute_script('scout_turnright_90.py')
    execute_script('scout_turnforward_50cm.py')
    execute_script('scout_turnleft_90.py')
    execute_script('scout_forward_3cm.py')
    read_serial_and_save(serial_port, os.path.join(base_path, '450_50.txt'))

    for i in range(8):
        execute_script('scout_forward_50cm.py')
        filename = os.path.join(base_path, '450_%d.txt' % (100 + i*50))
        read_serial_and_save(serial_port, filename)

if __name__ == '__main__':
    main()
