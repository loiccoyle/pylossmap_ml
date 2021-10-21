from pylossmap.utils import DB
import pytimber

if DB is None:
    try:
        DB = pytimber.LoggingDB()
    except Exception as exc:
        print("Failed to access pytimber logging.")
