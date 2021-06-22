from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from app import app
from app import dash_app1

# test wsgi
application = DispatcherMiddleware(
    app,
    {
        "/timeseries": dash_app1.app,
    },
)


run_simple("0.0.0.0", 8080, app, use_reloader=True, use_debugger=True)
