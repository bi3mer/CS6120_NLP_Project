from colorama import Fore, Style

def info(text):
    print(f"{Fore.LIGHTBLUE_EX}INFO:{Style.RESET_ALL} {text}")

def warning(text):
    print(f"{Fore.YELLOW}WARNING:{Style.RESET_ALL} {text}")

def error(text):
    print(f"{Fore.RED}ERROR:{Style.RESET_ALL} {text}")