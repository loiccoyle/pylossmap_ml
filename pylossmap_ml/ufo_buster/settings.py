# From Anton Lechner
# UFO buster algorithm:
# - Two BLMs within 40 m must exceed a certain dose rate threshold in RS04 (640us). This threshold was usually set to 1E-4 Gy/s and sometimes to 2E-4 Gy/s.
# - The ratio of RS02/RS01 must be larger than 0.55
# - The ratio of RS03/RS01 must be larger than 0.3

MAX_DISTANCE_RS04 = 4000  # cm
THRESHOLD_RS04 = 1e-4  # Gy/s
THRESHOLD_RS04_ALT = 2e-4  # Gy/s
THRESHOLD_RATIO_RS02_RS01 = 0.55
THRESHOLD_RATIO_RS03_RS01 = 0.3
