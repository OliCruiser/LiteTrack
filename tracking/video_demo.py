import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


def run_video(tracker_name, tracker_param, videofile=None, camera=None, epoch=100,
              camera_width=1920, camera_height=1080, camera_fps=None, camera_fourcc='MJPG',
              optional_box=None, debug=None, save_results=False):
    """Run the tracker on a video file or webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video", run_id=epoch)
    tracker.run_video(
        videofilepath=videofile,
        camera_id=camera,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_fps=camera_fps,
        camera_fourcc=camera_fourcc,
        optional_box=optional_box,
        debug=debug,
        save_results=save_results
    )


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on a video file or webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofile', type=str, nargs='?', default=None, help='Path to a video file.')
    parser.add_argument('--camera', type=int, default=None, help='Camera index, e.g. 0 for default webcam.')
    parser.add_argument('--epoch', type=int, default=100, help='Checkpoint epoch to load.')
    parser.add_argument('--camera_width', type=int, default=1920, help='Requested webcam width.')
    parser.add_argument('--camera_height', type=int, default=1080, help='Requested webcam height.')
    parser.add_argument('--camera_fps', type=float, default=None, help='Requested webcam FPS.')
    parser.add_argument('--camera_fourcc', type=str, default='MJPG',
                        help='Requested webcam FOURCC, e.g. MJPG. Use empty string to skip.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()
    if args.camera is None and args.videofile is None:
        parser.error('Please provide a videofile or set --camera.')

    run_video(
        args.tracker_name,
        args.tracker_param,
        args.videofile,
        args.camera,
        args.epoch,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
        args.camera_fourcc,
        args.optional_box,
        args.debug,
        args.save_results
    )


if __name__ == '__main__':
    main()
