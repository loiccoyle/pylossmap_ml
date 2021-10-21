import pytimber
from pylossmap.utils import DB

if DB is None:
    try:
        DB = pytimber.LoggingDB()
    except Exception as exc:
        print("Failed to access pytimber logging.")
