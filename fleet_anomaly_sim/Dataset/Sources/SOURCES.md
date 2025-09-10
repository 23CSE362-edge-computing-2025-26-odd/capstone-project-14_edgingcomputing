SIMULATED DATA
This project makes use of the IMS Bearing Dataset, which is a well-known dataset in machine condition monitoring. In the original experiment, four bearings were mounted on a shaft running at 2000 RPM under a constant heavy load, and vibration signals were collected using accelerometers. Each file in the dataset contains one second of vibration data (around 20,000 samples at 20 kHz), recorded every 10 minutes. Over time, the bearings naturally developed faults.

For the purpose of this work, only a single vibration channel was taken from the dataset. A small portion of the raw data was then used as a basis to simulate additional signals that represent both healthy and faulty machine behavior, demonstrate how the architechture works without having to process the entire dataset.

ACKNOWLEDGEMENTS
J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati Bearing Data Set, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA. Available at: https://data.nasa.gov/dataset/ims-bearings