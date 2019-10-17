#!/usr/bin/env python
 
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os

from Toxicity import Model
from Utility import log

class HTTPServer_RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(self.path)
            
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        message = '''
            <body>
                <title>
                    Hello!
                </title>

                Instead of serving content through writing in python, just write web files, read it, and serve it based on
                the self.path file. Easy :D

                <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
                <script>
                    // alert('requesting toxicity score for a sentence!');
                    $.post("rate/", {data: "example sentence!"}, (result, status) => {
                        alert(status + "---->" + result);
                    });
                </script>
            </body>
        '''

        self.wfile.write(bytes(message, "utf8"))
        return

    def do_POST(self):
        log.info(f'POST: {self.path}')
        response = {'completed': True}
        if '/rate' in self.path:
            model = Model()
            response = model.score('ah!')

        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.end_headers()
        self.wfile.write(bytes(json.dumps(response), 'utf8'))
        return
 
def run():
    print('starting server...')
    server_address = ('127.0.0.1', 8080)
    httpd = HTTPServer(server_address, HTTPServer_RequestHandler)
    httpd.serve_forever()

if __name__ == "__main__":
    run()

if __name__ == '__main__':
    log.info('logging info is working')

    log.warning('logging warning is working')
    log.error('logging error is working')

    model = Model()
    toxic_sentence = 'down with colan!'
    log.info(f'Random Score: {model.score(toxic_sentence)}')
