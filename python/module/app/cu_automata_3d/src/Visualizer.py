import pyvista as pv
from tqdm import tqdm
import cv2

class Visualizer(object):
    def __init__(self, states, idx=-1, width=1200, height=600, vista_kwargs={'cmap': 'viridis', 'opacity': [0, 0.1]}, plotter_kwargs={}):
        self.states = states
        self.vista_kwargs = vista_kwargs
        self.preview = pv.Plotter(**plotter_kwargs)
        self.preview.add_volume(states[idx], **self.vista_kwargs)
        self.size=(width, height)

    def Preview(self, show_kwargs={}):
        self.preview.show(**show_kwargs)
    
    def Render(self, path, fps):
        fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
        video=cv2.VideoWriter(path, fourcc, fps, self.size)
        for state in tqdm(self.states):
            pl = pv.Plotter(off_screen=True)
            pl.add_volume(state, **self.vista_kwargs)
            pl.camera_position = self.preview.camera_position
            img = pl.screenshot(filename=None, transparent_background=False, return_img=True, window_size=self.size)
            video.write(img[:,:,::-1])
        cv2.destroyAllWindows()
        video.release()
