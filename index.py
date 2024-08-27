import tornado.ioloop
import tornado.web
import argparse
from model import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--port', type=int, default=9263, help='Port to run the server on. Default: 9263')
argparser.add_argument('--model', type=str, default='model/classifier.model', help='Path to the model. Default: model/classifier.model')
args = argparser.parse_args()

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html", message=None)

class SubmitHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            data = self.get_body_argument("data")
            classifier = load_model(args.model)
            explainer = ShapExplainer(classifier)
            result = explainer.explain(data)
        except Exception as e:
            self.render("templates/index.html", message=f"An error occurred: {e}")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/submit", SubmitHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
