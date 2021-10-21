from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from pylossmap import BLMData, BLMDataFetcher
from pylossmap.lossmap import LossMap
from pylossmap.utils import sanitize_t


class Vizualizer:
    def __init__(
        self, model, data: pd.DataFrame, prediction: pd.Series, raw_data_dir: Path
    ):
        """Helper class for vizualization.

        Args:
            model (): ML model.
            data: The data the model was trained on.
            prediction: The model's predictions.
            raw_data_dir: Directory containing the raw data.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.model = model
        self.data = data
        if not isinstance(prediction, pd.Series):
            self.prediction = self._prep_predictions(prediction)
        else:
            self.prediction = prediction

    def _prep_predictions(self, prediction):
        if not isinstance(prediction, pd.Series):
            return pd.Series(prediction, index=self.data.index, name="prediction")
        else:
            return prediction

    @staticmethod
    def _defaults(kwargs):
        defaults = {}
        if "figsize" in kwargs.keys():
            defaults["figsize"] = kwargs["figsize"]
        else:
            defaults["figsize"] = (12, 8)

        return defaults

    def _blm_data_from_fill(self, fill_number):
        hdf_file = (self.raw_data_dir / str(fill_number)).with_suffix(".h5")
        if not hdf_file.exists():
            fetcher = BLMDataFetcher()
            return fetcher.from_fill(fill_number, beam_modes=["STABLE"])
            # raise ValueError(f"{hdf_file} does not exist.")

        return BLMData.load(hdf_file)

    def prediction_hist(self, **hist_kwargs):
        defaults = self._defaults(hist_kwargs)
        fig = plt.figure(figsize=defaults["figsize"])
        ax = plt.hist(self.prediction, **hist_kwargs)
        plt.grid()
        return fig, ax

    def waterfall(
        self, fill_number, waterfall_kwargs={}, prediction_kwargs={}, **kwargs
    ):
        defaults = self._defaults(kwargs)

        blm_data = self._blm_data_from_fill(fill_number)
        preprocessed_data = self.data.loc[fill_number]
        pred = self.prediction.loc[fill_number]

        fig, axs = plt.subplots(2, 2, figsize=defaults["figsize"], sharey=True)
        blm_data.plot(ax=axs[0][0], title="", **waterfall_kwargs)
        blm_data.plot(
            ax=axs[1][0],
            data=preprocessed_data,
            fill_missing=False,
            title="",
            **waterfall_kwargs,
        )

        pred_frame = pred.droplevel("mode").to_frame().reset_index()
        pred_frame["timestamp"] = pred_frame["timestamp"]

        pred_frame.plot(
            x="prediction", y="timestamp", ax=axs[0][1], **prediction_kwargs
        )
        # axs[0][1].invert_yaxis()
        pred_frame.plot(
            x="prediction", y="timestamp", ax=axs[1][1], **prediction_kwargs
        )
        plt.suptitle(fill_number)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig, axs

    def waterfall_animate(
        self,
        fill_number,
        filename=None,
        waterfall_kwargs={},
        lossmap_kwargs={},
        anim_kwargs={
            "interval": 16,
            "cache_frame_data": False,
            "save_count": 36000,
            "blit": True,
        },
        hline_kwargs={"color": "k"},
        skip_n=20,
        **kwargs,
    ):
        defaults = self._defaults(kwargs)

        if filename is None:
            filename = f"{fill_number}_waterfall.gif"
        filename = Path(filename)

        blm_data = self._blm_data_from_fill(fill_number)
        pred = self.prediction.loc[fill_number]

        fig = plt.figure(figsize=defaults["figsize"])
        gs = fig.add_gridspec(2, 2)
        # create the two top plots and single bottom plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[1, :])

        # plot the waterfall
        blm_data.plot(ax=ax1, title="", **waterfall_kwargs)
        hline1 = ax1.axhline(
            blm_data.df.index.get_level_values("timestamp")[0], **hline_kwargs
        )

        # plot the prediction
        pred_frame = pred.droplevel("mode").to_frame().reset_index()
        pred_frame["timestamp"] = pred_frame["timestamp"]
        pred_frame.plot(x="prediction", y="timestamp", ax=ax2)
        hline2 = ax2.axhline(
            blm_data.df.index.get_level_values("timestamp")[0], **hline_kwargs
        )

        # plot the initial lossmap
        fig, ax = blm_data[0].plot(ax=ax3, title=blm_data[0].datetime, **lossmap_kwargs)
        stem_containers = ax.containers
        title_artist = ax.title

        # animate the whole thing
        def update(i_lm):
            i = i_lm[0]
            lm = i_lm[1]
            hline1.set_ydata([lm.datetime, lm.datetime])
            hline2.set_ydata([lm.datetime, lm.datetime])
            title_artist.set_text(str(lm.datetime))
            for stem_c in stem_containers:
                blm_type = stem_c._label
                lm_type = lm.type(blm_type).df
                # duplicate
                lm_ar = np.repeat(lm_type[["dcum", "data"]].values, 2, axis=0)
                # set every second to 0
                lm_ar[1::2, :] = 0
                stem_c.stemlines.set_paths([lm_ar])
            print(i * skip_n)
            return [stem_c.stemlines for stem_c in stem_containers] + [
                title_artist,
                hline1,
                hline2,
            ]

        anim = animation.FuncAnimation(
            fig, update, frames=enumerate(skip_iter(blm_data, skip_n)), **anim_kwargs
        )

        # save the result
        anim.save(filename, dpi=80, writer="imagemagick")
        return fig, (ax1, ax2, ax3), anim

    def lossmap(self, time, lossmap_kwargs={}, **kwargs):
        defaults = self._defaults(kwargs)

        time = sanitize_t(time)
        row_index = self.data.index.get_level_values("timestamp").get_loc(
            time, method="nearest"
        )
        fill_number = self.data.index.get_level_values("fill_number")[row_index]
        blm_data = self._blm_data_from_fill(fill_number)

        fig, axs = plt.subplots(1, 1, figsize=defaults["figsize"], sharex=True)
        lm = blm_data[time]
        # plot the entire lossmap
        fig, ax = lm.normalize().IR(3, 4, 5, 6, 7).plot(ax=axs, **lossmap_kwargs)
        ax.set_yticks([10 ** p for p in range(-7, 2)])

        data_lm = pd.concat(
            [self.data.iloc[row_index], lm.meta],
            axis=1,
            join="inner",
        )
        data_lm.columns = ["data", "type", "dcum"]
        data_lm = LossMap(data_lm, time)
        # plot the lossmap as seen by the model
        # data_lm.plot(ax=axs[1],
        #              ylim=None,
        #              xlim=[0, lm.df['dcum'].max()], **lossmap_kwargs)
        # axs[1].set_xticks(data_lm.df['dcum'])
        # axs[1].set_xticklabels(data_lm.df.index,
        #                        rotation='vertical',
        #                        fontsize=6)

        # plt.suptitle(time.tz_convert('UTC'))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.legend(loc=8)
        return fig, axs

    def get_lossmap(self, time):
        time = sanitize_t(time)
        row_index = self.data.index.get_level_values("timestamp").get_loc(
            time, method="nearest"
        )
        fill_number = self.data.index.get_level_values("fill_number")[row_index]
        blm_data = self._blm_data_from_fill(fill_number)
        return blm_data[time]

    def lossmap_animate(
        self,
        fill_number,
        filename=None,
        ax=None,
        lossmap_kwargs={},
        anim_kwargs={
            "interval": 16,
            "cache_frame_data": False,
            "save_count": 36000,
            "blit": True,
        },
        skip_n=20,
        **kwargs,
    ):
        if filename is None:
            filename = f"{fill_number}_lossmap.gif"
        filename = Path(filename)

        defaults = self._defaults(kwargs)

        blm_data = self._blm_data_from_fill(fill_number)
        lm_init = blm_data[0]

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=defaults["figsize"])
        else:
            fig = ax.get_figure()

        # plot the initial lossmap
        fig, ax = lm_init.plot(ax=ax, **lossmap_kwargs)
        # print(ax.__dict__)
        print(len(blm_data.df.iloc[:, 0]))

        stem_containers = ax.containers
        title_artist = ax.title

        # # animate the lossmap
        def update(i_lm):
            i = i_lm[0]
            lm = i_lm[1]
            title_artist.set_text(str(lm.datetime))
            for stem_c in stem_containers:
                blm_type = stem_c._label
                lm_type = lm.type(blm_type).df
                # duplicate
                lm_ar = np.repeat(lm_type[["dcum", "data"]].values, 2, axis=0)
                # set every second to 0
                lm_ar[1::2, :] = 0
                stem_c.stemlines.set_paths([lm_ar])
            print(i * 1000)
            return [stem_c.stemlines for stem_c in stem_containers] + [title_artist]

        anim = animation.FuncAnimation(
            fig, update, frames=enumerate(skip_iter(blm_data, skip_n)), **anim_kwargs
        )
        anim.save(filename, dpi=80, writer="imagemagick")
        return fig, ax


def skip_iter(iterator, n):
    for i, element in enumerate(iterator):
        if i % n == 0:
            yield element
