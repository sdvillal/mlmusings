def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_immediate_subdirectories(dir):
    return [os.path.join(dir, name) for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]

def rm_tree(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

def username():
    import getpass
    getpass.getuser()

def file_to_str(src):
    if isinstance(src, str):
        return src
    return src.name