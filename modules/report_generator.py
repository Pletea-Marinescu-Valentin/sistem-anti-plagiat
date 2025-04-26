from datetime import datetime
import os

class ReportGenerator:
    def __init__(self, timestamp, save_path="./reports"):
        self.timestamp = timestamp
        # folosim calea din configuratie
        self.save_path = save_path

    def generate_html_report(self, violation_log, output_video_path):
        # construim calea completa
        report_filename = f"raport_anti_plagiat_{self.timestamp}.html"
        report_path = os.path.join(self.save_path, report_filename)

        # cream directorul daca nu exista
        os.makedirs(self.save_path, exist_ok=True)

        total_violations = len(violation_log)
        grouped_violations = {}

        for record in violation_log:
            for violation in record["violations"]:
                violation_type = violation.split(":")[0].strip()
                if violation_type not in grouped_violations:
                    grouped_violations[violation_type] = 0
                grouped_violations[violation_type] += 1

        # generare html
        html_content = f"""<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raport Anti-Plagiat</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2 {{
            color: #2c3e50;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .summary {{
            background-color: #e9f7ef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .chart {{
            margin: 30px 0;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }}
        .violation {{
            color: #e74c3c;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Raport Sistem Anti-Plagiat</h1>
        <p>Raport generat la: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <h2>Sumar</h2>
            <p>Total incalcari detectate: <strong>{total_violations}</strong></p>
            <p>Durata monitorizare: De la {violation_log[0]["timestamp"] if violation_log else "N/A"} pana la {violation_log[-1]["timestamp"] if violation_log else "N/A"}</p>
            <p>Inregistrare video salvata in: <strong>{output_video_path}</strong></p>
        </div>

        <div class="chart">
            <h2>Tipuri de incalcari</h2>
            <table>
                <tr>
                    <th>Tipul incalcarii</th>
                    <th>Numar de aparitii</th>
                </tr>
                {"".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in grouped_violations.items()])}
            </table>
        </div>

        <h2>Detalii incalcari</h2>
        <table>
            <tr>
                <th>Timestamp</th>
                <th>Incalcari detectate</th>
            </tr>
            {"".join([f"<tr><td>{record['timestamp']}</td><td class='violation'>{', '.join(record['violations'])}</td></tr>" for record in violation_log])}
        </table>
    </div>
</body>
</html>
"""

        # salvare raport html
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Raportul HTML a fost generat: {report_path}")
        return report_path