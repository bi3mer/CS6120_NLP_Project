#!/usr/bin/env python
 
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os

from Toxicity.Utility import log
from Toxicity import Model

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
        #log.info(f'POST: {self.path}')

        '''
        There are two ways you can go about this. You can either get first receive the user name
        and send back the tweets associated with them. The webpage can then send each tweet and
        receive a score. The other way is to send the user name and then have the server retrieve
        the tweets and then call the model.score method for each tweet. You then calculate the 
        average and you're done. The potential issue with this second approach is total calculation
        time that may result in timeouts. This is possible to avoid, you just need to get your
        heroku and webpage settings correct.
        '''

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
    # log.info('starting server...')
    server_address = ('127.0.0.1', 8080)
    httpd = HTTPServer(server_address, HTTPServer_RequestHandler)
    httpd.serve_forever()
    # log.info(f'servering on port: 8080')

if __name__ == "__main__":
    run()
