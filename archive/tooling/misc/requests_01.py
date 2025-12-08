#! /usr/bin/python3

from getpass import getpass 
import requests
import argparse


def run_query(username):

    # password = getpass("Password for 'https://{}@github.com'".format(username))
    password = "amdG1i1t1"
    
    response = requests.get('https://api.github.com/user', auth=(username, password))
    print (response.headers)
    print("")
    print (response.json())


if __name__ == "__main__" :

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-u", "--username", required=True)
    # args = parser.parse_args()

    run_query("deven-amd")
