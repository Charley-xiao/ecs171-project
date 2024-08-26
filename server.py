import tornado.ioloop
import tornado.web
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--port', type=int, default=9263, help='Port to run the server on. Default: 9263')
args = argparser.parse_args()

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html", message=None)

class SubmitHandler(tornado.web.RequestHandler):
    def post(self):
        data = self.get_body_argument("data")
        self.render("templates/index.html", message=f"Received: {data}")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/submit", SubmitHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
