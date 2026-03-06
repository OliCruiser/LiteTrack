import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

# from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, debug=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None
        params = self.get_parameters()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        self.params = params
        # self.create_tracker(params)

    def create_tracker(self, params):
        self.tracker = self.tracker_class(params, self.dataset_name)

    def run_sequence(self, seq, debug=None, vis=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        # Get init information
        self.create_tracker(self.params)
        init_info = seq.init_info()
        output = self._track_sequence( seq, init_info, vis)
        out = output.copy()
        del self.tracker
        return out

    def _track_sequence(self, seq, init_info, vis=None):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if self.tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = self.tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if self.tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = self.tracker.track(image, info, vis)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath=None, camera_id=None, camera_width=1920, camera_height=1080,
                  camera_fps=None, camera_fourcc='MJPG', optional_box=None, debug=None,
                  visdom_info=None, save_results=False):
        """Run the tracker with a video file or webcam.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            self.create_tracker(params)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        output_boxes = []
        if camera_id is None:
            assert videofilepath is not None and os.path.isfile(videofilepath), \
                "videofilepath must be a valid videofile"
            cap = cv.VideoCapture(videofilepath)
            save_name = Path(videofilepath).stem
        else:
            cap = cv.VideoCapture(camera_id)
            if camera_fourcc is not None and camera_fourcc != '':
                assert len(camera_fourcc) == 4, "camera_fourcc must be a 4-char code, e.g. MJPG"
                cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*camera_fourcc))
            if camera_width is not None:
                cap.set(cv.CAP_PROP_FRAME_WIDTH, int(camera_width))
            if camera_height is not None:
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(camera_height))
            if camera_fps is not None:
                cap.set(cv.CAP_PROP_FPS, float(camera_fps))
            save_name = 'camera_{}_{}'.format(camera_id, int(time.time()))

        if not cap.isOpened():
            raise RuntimeError('Cannot open {}'.format(
                videofilepath if camera_id is None else 'camera {}'.format(camera_id)))
        if camera_id is not None:
            actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv.CAP_PROP_FPS)
            print('Camera {} requested {}x{}{}{}; actual {}x{} @ {:.2f} FPS'.format(
                camera_id,
                camera_width if camera_width is not None else 'default',
                camera_height if camera_height is not None else 'default',
                '' if camera_fps is None else ' @ ',
                '' if camera_fps is None else '{} FPS'.format(camera_fps),
                actual_width, actual_height, actual_fps))

        display_name = 'Display: ' + self.tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)

        def _build_init_info(box):
            return {'init_bbox': box}

        def _read_frame_with_retry(max_retry=5, sleep_s=0.005):
            retries = max_retry if camera_id is not None else 1
            for _ in range(retries):
                try:
                    ret, read_frame = cap.read()
                except cv.error:
                    ret, read_frame = False, None
                if ret and read_frame is not None and read_frame.size > 0:
                    return True, read_frame
                if camera_id is not None:
                    time.sleep(sleep_s)
            return False, None

        def _select_roi_on_frame(frame):
            frame_disp = frame.copy()
            cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       1.5, (0, 0, 0), 1)
            x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            return [int(x), int(y), int(w), int(h)]

        def _wait_for_camera_roi():
            while True:
                ret, live_frame = _read_frame_with_retry(max_retry=8)
                if not ret or live_frame is None:
                    key = cv.waitKey(1) & 0xFF
                    if key == ord('q'):
                        return None, None, True
                    continue

                frame_disp = live_frame.copy()
                cv.putText(frame_disp, 'Press SPACE/f to freeze', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (0, 0, 0), 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (0, 0, 0), 1)
                cv.imshow(display_name, frame_disp)
                key = cv.waitKey(1) & 0xFF

                if key == ord('q'):
                    return None, None, True
                if key in (ord(' '), ord('f')):
                    init_state = _select_roi_on_frame(live_frame)
                    if init_state[2] > 0 and init_state[3] > 0:
                        return live_frame, init_state, False

        frame = None
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            success, frame = _read_frame_with_retry(max_retry=8)
            if not success or frame is None:
                raise RuntimeError("Read frame failed.")
            self.tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        elif camera_id is None:
            success, frame = _read_frame_with_retry(max_retry=8)
            if not success or frame is None:
                raise RuntimeError("Read frame from {} failed.".format(videofilepath))
            cv.imshow(display_name, frame)
            init_state = _select_roi_on_frame(frame)
            if init_state[2] <= 0 or init_state[3] <= 0:
                cap.release()
                cv.destroyAllWindows()
                return
            self.tracker.initialize(frame, _build_init_info(init_state))
            output_boxes.append(init_state)
        else:
            frame, init_state, should_quit = _wait_for_camera_roi()
            if should_quit:
                cap.release()
                cv.destroyAllWindows()
                return
            self.tracker.initialize(frame, _build_init_info(init_state))
            output_boxes.append(init_state)

        prev_frame_time = time.time()
        smooth_fps = 0.0

        while True:
            ret, frame = _read_frame_with_retry(max_retry=3)

            if not ret or frame is None:
                if camera_id is None:
                    break
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue

            frame_disp = frame.copy()

            now = time.time()
            dt = now - prev_frame_time
            prev_frame_time = now
            instant_fps = 1.0 / dt if dt > 0 else 0.0
            smooth_fps = instant_fps if smooth_fps == 0.0 else 0.9 * smooth_fps + 0.1 * instant_fps

            # Draw box
            out = self.tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (57, 255, 20)
            font_scale = 1.2
            font_thickness = 2
            cv.putText(frame_disp, 'FPS: {:.2f}'.format(smooth_fps), (20, 35), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       font_scale, font_color, font_thickness)
            cv.putText(frame_disp, 'Tracking!', (20, 65), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       font_scale, font_color, font_thickness)
            cv.putText(frame_disp, 'Press r to reset', (20, 95), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       font_scale, font_color, font_thickness)
            cv.putText(frame_disp, 'Press q to quit', (20, 125), cv.FONT_HERSHEY_COMPLEX_SMALL,
                       font_scale, font_color, font_thickness)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if camera_id is None:
                    ret, frame = _read_frame_with_retry(max_retry=8)
                    if not ret or frame is None:
                        break
                    init_state = _select_roi_on_frame(frame)
                    if init_state[2] <= 0 or init_state[3] <= 0:
                        continue
                else:
                    frame, init_state, should_quit = _wait_for_camera_roi()
                    if should_quit:
                        break
                self.tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(save_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name, self.run_id)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        # elif isinstance(image_file, list) and len(image_file) == 2:
        #     return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")
