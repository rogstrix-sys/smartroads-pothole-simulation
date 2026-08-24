from vehicle import Driver

driver = Driver()

TIME_STEP = int(driver.getBasicTimeStep())

# Start completely stopped
driver.setCruisingSpeed(0.0)

while driver.step() != -1:
    pass