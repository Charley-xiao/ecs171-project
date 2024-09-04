import tornado.ioloop
import tornado.web
import argparse
import json 
from model import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--port', type=int, default=9263, help='Port to run the server on. Default: 9263')
argparser.add_argument('--model', type=str, default='model/classifier.model', help='Path to the model. Default: model/classifier.model')
argparser.add_argument('--label2ind', type=str, default='data/label2ind.json', help='Path to the label2ind file.')
args = argparser.parse_args()

def get_name_by_id(author_id):
    for name, id in label2ind.items():
        if int(id.split('__')[-1]) == author_id:
            return name

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html", message=None)

class SubmitHandler(tornado.web.RequestHandler):
    def post(self):
        try:
            data = self.get_body_argument("data")
            explainer = ShapExplainer(classifier)
            data = data.replace('\n', ' ')
            print(f"Explaining instance:\n {data}")
            result_model_list = predict(data, classifier, args.label2ind)
            result_model_list.sort(key=lambda x: x[2], reverse=True)
            show_result = [(x[1], round(x[2], 2)) for x in result_model_list][:3]
            top3_author_ids = [int(result_model[0]) for result_model in result_model_list][:3]
            print(f"Top 3 authors: {top3_author_ids}")
            result_sentence, result_dict = explainer.explain(data, top3_author_ids)
            segments = []
            for start_end, color, top_3_indexes in result_dict.values():
                start, end = start_end
                text = result_sentence[start:end]
                segments.append((color, text, (get_name_by_id(top_3_indexes[0][0]), round(top_3_indexes[0][1], 2))))
            self.render("templates/submit.html", segments=segments, show_result=show_result)
        except tornado.web.MissingArgumentError:
            self.render("templates/index.html", message="Please enter a text.")
        except AttributeError:
            self.render("templates/index.html", message="Please enter at least 2 sentences.")
        except IndexError:
            self.render("templates/index.html", message="Your text is too long!")
        except Exception as e:
            self.render("templates/index.html", message=f"An error occurred: {e}")

class FavoriteIconHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header('Content-Type', 'image/png')
        with open('templates/favicon.png', 'rb') as f:
            self.write(f.read())

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/submit", SubmitHandler),
        (r"/favicon.png", FavoriteIconHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(args.port)
    print(f"Server running on http://localhost:{args.port}/ with model {args.model}")
    classifier = load_model(args.model)
    label2ind = json.load(open(args.label2ind, 'r', encoding='ISO-8859-1'))
    tornado.ioloop.IOLoop.current().start()