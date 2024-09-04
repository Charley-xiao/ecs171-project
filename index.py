import tornado.ioloop
import tornado.web
import concurrent.futures
from tornado.concurrent import run_on_executor
from tornado.ioloop import IOLoop
from tornado.web import RequestHandler
import os 
import argparse
import json 
from model import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--port', type=int, default=9263, help='Port to run the server on. Default: 9263')
argparser.add_argument('--model', type=str, default='model/classifier.model', help='Path to the model. Default: model/classifier.model')
argparser.add_argument('--label2ind', type=str, default='data/label2ind.json', help='Path to the label2ind file.')
args = argparser.parse_args()

classifier = load_model(args.model)
label2ind = json.load(open(args.label2ind, 'r', encoding='ISO-8859-1'))

max_workers = os.cpu_count() * 2

representative_works = {
    'Charles Dickens': ['A Tale of Two Cities', 'Great Expectations', 'Oliver Twist'],
    'Agatha Christie': ['The Mysterious Affair at Styles'],
    'Jane Austen': ['Pride and Prejudice', 'Sense and Sensibility', 'Emma'],
    'Mark Twain': ['The Adventures of Tom Sawyer', 'Adventures of Huckleberry Finn'],
    'O Henry': ['The Gift of the Magi', 'The Ransom of Red Chief'],
    'Oscar Wilde': ['The Picture of Dorian Gray', 'The Importance of Being Earnest'],
    'P G Wodehouse': ['Right Ho, Jeeves', 'The Inimitable Jeeves'],
    'Walt Whitman': ['Leaves of Grass'],
    'Winston Churchill': ['The Crisis', 'The Crossing'],
    'Zane Grey': ['Riders of the Purple Sage', 'The Last Trail']
}

def get_name_by_id(author_id):
    for name, id in label2ind.items():
        if int(id.split('__')[-1]) == author_id:
            return name

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("templates/index.html", message=None)

class SubmitHandler(RequestHandler):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    @run_on_executor
    def compute_prediction(self, data):
        explainer = ShapExplainer(classifier)
        data = data.replace('\n', ' ')
        result_model_list = predict(data, classifier, args.label2ind)
        result_model_list.sort(key=lambda x: x[2], reverse=True)
        show_result = [(x[1], round(x[2], 2), representative_works[x[1]]) for x in result_model_list[:3]]
        top3_author_ids = [int(result_model[0]) for result_model in result_model_list[:3]]
        result_sentence, result_dict = explainer.explain(data, top3_author_ids)
        
        segments = []
        for start_end, color, top_3_indexes in result_dict.values():
            start, end = start_end
            text = result_sentence[start:end]
            author_name = get_name_by_id(top_3_indexes[0][0])
            segments.append((color, text, (author_name, round(top_3_indexes[0][1], 2))))

        return segments, show_result

    async def post(self):
        try:
            data = self.get_body_argument("data")
            # print(f"Explaining instance:\n {data}")

            segments, show_result = await self.compute_prediction(data)

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
    ], compress_response=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(args.port)
    print(f"Server running on http://localhost:{args.port}/ with model {args.model}")
    tornado.ioloop.IOLoop.current().start()