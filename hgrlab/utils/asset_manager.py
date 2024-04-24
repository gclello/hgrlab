import os
import tempfile
import urllib.request

class AssetManager:
    '''Remote assets catalog manager'''
    
    def __init__(self):
        self.remote_assets = {}
        self.local_assets = {}
        
    def __str__(self):
        return '[AssetManager] Remote assets: %d | Local assets: %d' % (
            len(self.remote_assets.keys()),
            len(self.local_assets.keys()),
        )
    
    def add_remote_asset(self, key, remote_id, filename):
        self.remote_assets[key] = {
            "remote_id": remote_id,
            "filename": filename,
        }

    def get_asset_url(self, key):
        remote_id = self.remote_assets[key]['remote_id']
        return 'https://drive.google.com/uc?export=download&id=%s' % remote_id
    
    @staticmethod
    def get_base_dir():
        return os.path.join(tempfile.gettempdir(), 'hgrlab')
    
    def get_asset_path(self, key):
        filename = self.remote_assets[key]['filename']
        base_dir = AssetManager.get_base_dir()
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        return os.path.join(base_dir, filename)
    
    def download_asset(self, key):
        path = self.get_asset_path(key)
        url = self.get_asset_url(key)
        cached = False

        if not os.path.isfile(path):
            urllib.request.urlretrieve(
                url,
                path,
            )
        else:
            cached = True

        if os.path.isfile(path):
            self.local_assets[key] = self.remote_assets[key]
        
        return cached
