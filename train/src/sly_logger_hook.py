import datetime
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook
import supervisely_lib as sly
from sly_train_progress import get_progress_cb, set_progress, add_progress_to_request
import sly_globals as g
import classes as cls


@HOOKS.register_module()
class SuperviselyLoggerHook(TextLoggerHook):
    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 interval_exp_name=1000):
        super(SuperviselyLoggerHook, self).__init__(by_epoch, interval, ignore_last, reset_flag, interval_exp_name)
        self.progress_epoch = None
        self.progress_iter = None
        self._lrs = []

    def _log_info(self, log_dict, runner):
        if log_dict['mode'] == 'train' and 'time' in log_dict.keys():
            self.time_sec_tot += (log_dict['time'] * self.interval)
            time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (len(runner.data_loader) * runner.max_epochs - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_dict['sly_eta'] = eta_str

        if self.progress_epoch is None:
            self.progress_epoch = sly.Progress("Epochs", runner.max_epochs)
        if self.progress_iter is None:
            self.progress_iter = sly.Progress("Iterations", len(runner.data_loader))

        fields = []
        if log_dict['mode'] == 'train':
            self.progress_epoch.set_current_value(log_dict["epoch"])
            self.progress_iter.set(log_dict['iter'] % len(runner.data_loader), len(runner.data_loader))
            fields.append({"field": "data.eta", "payload": log_dict['sly_eta']})
            is_val = log_dict['iter'] % len(runner.data_loader) == 0
            fields.append({"field": "state.isValidation", "payload": is_val})
        else:
            fields.append({"field": "state.isValidation", "payload": True})

        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)

        epoch_float = \
            float(self.progress_epoch.current) + float(self.progress_iter.current) / float(self.progress_iter.total)
        if log_dict['mode'] == 'train':
            fields.extend([
                {"field": "state.chartLR.series[0].data", "payload": [[epoch_float, round(log_dict["lr"], 6)]], "append": True},
                {"field": "state.chartTrainLoss.series[0].data", "payload": [[epoch_float, round(log_dict["loss"], 6)]],
                 "append": True},
            ])
            '''
            self._lrs.append(log_dict["lr"])
            fields.append({
                "field": "state.chartLR.options.yaxisInterval",
                "payload": [
                    round(min(self._lrs) - min(self._lrs) / 10.0, 5),
                    round(max(self._lrs) + max(self._lrs) / 10.0, 5)
                ]
            })
            '''

            if 'time' in log_dict.keys():
                fields.extend([
                    {"field": "state.chartTime.series[0].data", "payload": [[epoch_float, log_dict["time"]]],
                     "append": True},
                    {"field": "state.chartDataTime.series[0].data", "payload": [[epoch_float, log_dict["data_time"]]],
                     "append": True},
                    {"field": "state.chartMemory.series[0].data", "payload": [[epoch_float, log_dict["memory"]]],
                     "append": True},
                ])
        if log_dict['mode'] == 'val':
            class_metrics = {}
            for field_name, field_val in log_dict.items():
                field_name_parts = field_name.split('.')
                if len(field_name_parts) == 1:
                    continue
                metric, class_name = field_name_parts[0], field_name_parts[1]

                if metric not in class_metrics.keys():
                    class_metrics[metric] = {}
                class_metrics[metric][class_name] = [[log_dict["epoch"], field_val]]
            for metric_name, metrics in class_metrics.items():
                if "m" + metric_name not in g.evalMetrics:
                    continue
                classes = cls.selected_classes + ["__bg__"]
                for class_ind, class_name in enumerate(classes):
                    fields.extend([
                        {"field": f"state.class_charts.chartVal_{metric_name}.series[{class_ind}].data",
                         "payload": metrics[class_name], "append": True}
                    ])
            for metric_name in g.evalMetrics:
                metric_values = [metric_val[0][1] for metric_val in list(class_metrics[metric_name[1:]].values())]
                mean_metric = sum(metric_values) / len(metric_values)
                fields.extend([
                    {"field": f"state.mean_charts.chartVal_{metric_name}.series[0].data",
                     "payload": [[log_dict["epoch"], mean_metric]], "append": True}
                ])
        try:
            g.api.app.set_fields(g.task_id, fields)
        except Exception as e:
            print("Unabled to write metrics to chart!")
            print(e)
