import visdom


class Visualizer(object):

    def __init__(self, display_port):
        self.vis = visdom.Visdom(port=display_port)

    def displayImg(self, imgSat, imgOut, imgLabel):
        self.vis.image(imgSat, opts=dict(title='SateImg'),
                       win='Sate')
        self.vis.image(imgOut, opts=dict(title='Output'),
                       win='Out')
        self.vis.image(imgLabel, opts=dict(title='LabelImg'),
                       win='Label')
        return

    def drawLine(self, X_val, Y_val):
        self.vis.line(X=X_val, Y=Y_val, win='line', update='append' if self.vis.win_exists('line') else None)
        return