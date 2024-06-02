from assembly import Assembly
import matplotlib.pyplot as plt


class Assembly3D(Assembly):
    def get_fig_ax(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #SIZE = max(self.rs) * 1.02
        #ax.set_xlim([-SIZE, SIZE])
        #ax.set_ylim([-SIZE, SIZE])
        #ax.set_aspect('equal')
        return fig, ax
