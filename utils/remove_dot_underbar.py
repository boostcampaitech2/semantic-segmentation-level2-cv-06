import os

def remove_dot_underbar(root):
    for path in os.listdir(root):
        tmp = os.path.join(root,path)
        if os.path.isdir(tmp):
            remove_dot_underbar(tmp)
        if path[0] == '.' and path[1] =='_':
            os.remove(tmp)
            print(tmp, 'is removed')


root = './'

remove_dot_underbar(root)