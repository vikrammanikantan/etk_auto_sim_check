import smtplib
from email.message import EmailMessage
from pathlib import Path
import os
import plotting

from email_config import (
    SMTP_SERVER,
    SMTP_PORT,
    SENDER_EMAIL,
    SENDER_PASSWORD,
    RECIPIENT_EMAIL,
)

import sys

sim_name = sys.argv[1]
output_num = sys.argv[2]

plotting.plot_simulation(sim_name, output_num)

# -------- USER INPUT ----------
RECIPIENT_EMAIL = RECIPIENT_EMAIL
PDF_PATH = "%s_%s.pdf"%(sim_name, str(output_num).zfill(4))
SUBJECT = "%s, output-%s latest"%(sim_name, str(output_num).zfill(4))
BODY = "%s, output-%s latest"%(sim_name, str(output_num).zfill(4))
# ------------------------------

pdf_path = Path(PDF_PATH)
if not pdf_path.exists():
    raise FileNotFoundError(f"{PDF_PATH} not found")

# Create email
msg = EmailMessage()
msg["From"] = SENDER_EMAIL
msg["To"] = RECIPIENT_EMAIL
msg["Subject"] = SUBJECT
msg.set_content(BODY)

# Attach PDF
with open(pdf_path, "rb") as f:
    msg.add_attachment(
        f.read(),
        maintype="application",
        subtype="pdf",
        filename=pdf_path.name,
    )

# Send email
with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)

# Delete PDF after successful send
pdf_path.unlink()

print(f"Email sent and deleted: {pdf_path}")
