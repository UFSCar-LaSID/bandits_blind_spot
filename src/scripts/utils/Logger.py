from datetime import datetime
import os

from src import FILE_LOGS

class Logger:
    '''
    Classe responsável por logar mensagens em um arquivo de log. Útil para debugar e verificar tempo de execução de diferentes partes do código, por exemplo.
    '''
    
    def __init__(self, log_file_path: str):
        '''
        Construtor da classe Logger.

        params:
            log_file_path: Caminho para o arquivo de log.
        '''
        os.makedirs(log_file_path, exist_ok=True)
        logger_file = os.path.join(log_file_path, FILE_LOGS)
        if os.path.exists(logger_file):
            raise ValueError('The log file {} already exists. Please choose another path.'.format(logger_file))
        self.logger = open(logger_file, 'w')
        

    def log(self, message: str, force_save: bool = False):
        '''
        Loga uma mensagem no arquivo de log.
        A mensagem é escrita no formato: `[YYYY-MM-DD HH:MM:SS] <message>`
        Cada mensagem é concatenada com uma nova linha.

        params:
            message: Mensagem a ser logada
            force_save: Se True, força a escrita do arquivo de log.
        '''
        self.logger.write('{} {}\n'.format(datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'), message))
        if force_save:
            self.__save()


    def print_and_log(self, message: str, force_save: bool = False):
        '''
        Printa a mensagem no terminal e loga a mensagem no arquivo de log.
        A mensagem é escrita no formato: `[YYYY-MM-DD HH:MM:SS] <message>`
        Cada mensagem é concatenada com uma nova linha.

        params:
            message: Mensagem a ser printada e logada
            force_save: Se True, força a escrita do arquivo de log.
        '''
        print('{} {}'.format(datetime.now().strftime('[%Y-%m-%d %H:%M:%S]'), message))
        self.log(message, force_save)


    def __save(self):
        '''
        Força a escrita do arquivo de log.
        '''
        self.logger.flush()
        os.fsync(self.logger.fileno())


    def __del__(self):
        '''
        Fecha o arquivo de log quando o objeto Logger é destruído.
        '''
        if hasattr(self, 'logger'):
            self.logger.close()


def format_elapsed_time(start, end):
    '''
    Formata o tempo decorrido entre `start` e `end` em horas, minutos e segundos.

    params:
        start: Tempo inicial.
        end: Tempo final.

    returns:
        String formatada com o tempo decorrido.
    '''
    elapsed_time = end - start
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:.0f}h {:.0f}m {:.0f}s'.format(hours, minutes, seconds)