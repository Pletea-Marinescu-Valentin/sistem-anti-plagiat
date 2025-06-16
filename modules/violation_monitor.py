from datetime import datetime

class ViolationMonitor:
    def __init__(self):
        self.violation_log = []

        # alert messages
        self.direction_messages = {
            "left": "Suspicious gaze to the LEFT",
            "right": "Suspicious gaze to the RIGHT", 
            "down": "Suspicious gaze DOWNWARD",
            "no_face": "Face is not visible"
        }

        # time of last recorded violation
        self.last_log_time = 0
        # minimum interval between recordings in seconds
        self.min_log_interval = 1.5

    def check_violations(self, direction, objects):
        violations = []

        # check gaze direction
        if direction in self.direction_messages:
            message = self.direction_messages[direction]
            violations.append(message)

        # Check object violations
        for obj_data in objects:
            if len(obj_data) >= 2:
                obj_type = obj_data[0]
                confidence = obj_data[1]
                
                if obj_type == 'phone':
                    violations.append(f"Phone detected (confidence: {confidence:.2f})")
                elif obj_type == 'smartwatch':
                    violations.append(f"Smartwatch detected (confidence: {confidence:.2f})")
            else:
                print(f"DEBUG: Unexpected object data format: {obj_data}")

        return violations

    def log_violation(self, violations):
        if violations:
            # get current time
            current_time = datetime.now()
            current_timestamp = current_time.timestamp()

            # check if minimum interval has passed since last recording
            time_diff = current_timestamp - self.last_log_time

            if time_diff >= self.min_log_interval:
                # record alert
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                violation_record = {
                    "timestamp": timestamp,
                    "violations": violations
                }
                self.violation_log.append(violation_record)
                self.last_log_time = current_timestamp
                return True
        return False

    def get_logs(self):
        return self.violation_log