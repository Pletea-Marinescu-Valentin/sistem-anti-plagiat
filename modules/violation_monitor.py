from datetime import datetime

class ViolationMonitor:
    def __init__(self):
        self.violation_log = []

        # mesaje pentru alerte
        self.direction_messages = {
            "left": "Privire suspecta spre STANGA",
            "right": "Privire suspecta spre DREAPTA",
            "down": "Privire suspecta in JOS",
            "no_face": "Fata nu este vizibila"
        }

        # timpul ultimei incalcari inregistrate
        self.last_log_time = 0
        # intervalul minim intre inregistrari in secunde
        self.min_log_interval = 1.5

    def check_violations(self, direction, objects):
        violations = []

        # verificam directia privirii
        if direction in self.direction_messages:
            message = self.direction_messages[direction]
            violations.append(message)

        # verificam obiectele detectate
        for obj, _ in objects:
            message = f"Obiect neautorizat detectat: {obj.upper()}"
            violations.append(message)

        return violations

    def log_violation(self, violations):
        if violations:
            # obtinem timpul curent
            current_time = datetime.now()
            current_timestamp = current_time.timestamp()

            # verificam daca a trecut intervalul minim de la ultima inregistrare
            time_diff = current_timestamp - self.last_log_time

            if time_diff >= self.min_log_interval:
                # inregistram alerta
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