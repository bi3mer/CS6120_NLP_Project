from Utility import log
from Toxicity import Model


if __name__ == '__main__':
    log.info('logging info is working')
    log.warning('logging warning is working')
    log.error('logging error is working')

    model = Model()
    toxic_sentence = 'down with colan!'
    log.info(f'Random Score: {model.score(toxic_sentence)}')
