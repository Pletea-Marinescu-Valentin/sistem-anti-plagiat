import csv
import json
import os
from datetime import datetime

class DataExporter:
    def __init__(self, config):
        self.config = config
        self.export_path = config["reporting"]["save_path"]
        self.formats = config["reporting"]["export_formats"]

        # ensure export directory exists
        os.makedirs(self.export_path, exist_ok=True)

    def export_data(self, violation_log, timestamp):
        """exports data in formats specified in configuration"""
        results = {}

        if "csv" in self.formats:
            csv_path = self._export_to_csv(violation_log, timestamp)
            results["csv"] = csv_path

        if "json" in self.formats:
            json_path = self._export_to_json(violation_log, timestamp)
            results["json"] = json_path

        return results

    def _export_to_csv(self, violation_log, timestamp):
        """exports data to CSV format"""
        csv_path = os.path.join(self.export_path, f"violations_{timestamp}.csv")

        # create dictionary for statistics
        violation_types = {}
        for record in violation_log:
            for violation in record["violations"]:
                violation_type = violation.split(":")[0].strip()
                if violation_type not in violation_types:
                    violation_types[violation_type] = 0
                violation_types[violation_type] += 1

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            # first write summary data
            writer = csv.writer(csvfile)
            writer.writerow(["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow(["Total Violations", str(len(violation_log))])
            writer.writerow([])

            # write statistics by violation types
            writer.writerow(["Violation Type", "Count"])
            for violation_type, count in violation_types.items():
                writer.writerow([violation_type, count])
            writer.writerow([])

            # write detailed violation records
            writer.writerow(["Timestamp", "Violations"])
            for record in violation_log:
                writer.writerow([record["timestamp"], ", ".join(record["violations"])])
            writer.writerow([])

        return csv_path

    def _export_to_json(self, violation_log, timestamp):
        """exports data to JSON format"""
        json_path = os.path.join(self.export_path, f"violations_{timestamp}.json")

        # create dictionary for statistics
        violation_types = {}
        for record in violation_log:
            for violation in record["violations"]:
                violation_type = violation.split(":")[0].strip()
                if violation_type not in violation_types:
                    violation_types[violation_type] = 0
                violation_types[violation_type] += 1

        export_data = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_violations": len(violation_log)
            },
            "statistics": {
                "violation_types": violation_types
            },
            "violations": violation_log
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=4)

        return json_path