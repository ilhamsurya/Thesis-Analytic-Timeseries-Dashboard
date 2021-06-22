from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from app import server
from app import dash_app1

application = DispatcherMiddleware(
    server,
    {
        "/timeseries": dash_app1.server,
    },
)


run_simple("0.0.0.0", 8080, server, use_reloader=True, use_debugger=True)
