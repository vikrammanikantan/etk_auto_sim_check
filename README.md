# Automatic Simulation Check
This code will plot key diagnostics from your simulation, save them to a PDF, email that PDF to you, and then will delete the PDF. The idea is to run this after every job/output to make sure the simulation is running properly.

You will have to do 2 things:
1. Copy email_config.py.temp to email_config.py and fill the variables with your own email address, app password (google this for each email provider), and recipient email
2. Adjust the plot_simulation function in plotting.py to your liking. (it might be easier to just create your own plotting.py that outputs a PDF and go from there)