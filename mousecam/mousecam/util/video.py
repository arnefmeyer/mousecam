
def get_video_files(path, extensions=['.mp4', '.h264']):

    video_files = []

    if op.isfile(path) and op.splitext(path)[1] in extensions:
        video_files = [path]

    elif op.isdir(path):

        video_files = []
        for root, dirs, files in os.walk(path):

            for f in files:

                if op.splitext(f)[1] in extensions:
                    video_files.append(op.join(root, f))

    return video_files


def get_first_frame(file_path, grayscale=True):

    import imageio
    import cv2

    with imageio.get_reader(file_path, 'ffmpeg') as reader:

        frame = reader.get_data(0)

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return frame


def dump_video_to_memmap_file(file_path, output=None, overwrite=False,
                              bbox=None, max_num_frames=-1):

    # memmap file path
    if output is None:
        mmap_file = op.splitext(file_path)[0] + '.memmap'
    else:
        mmap_file = op.join(output,
                            op.splitext(op.basename(file_path))[0] + '.memmap')

    param_file = mmap_file + '.params'

    if not op.exists(mmap_file) or overwrite:

        # get parameters
        rec_path, file_name = op.split(file_path)
        pattern = op.splitext(file_name)[0]
        params = rpicam.load_video_parameters(rec_path, pattern=pattern)

        if params['timestamps'] is None:
            # old database format
            params['timestamps'] = load_old_camera_timestamps(file_path)

        # open reader
        reader = imageio.get_reader(file_path, 'ffmpeg')

        # process frames
        fp = None
        size = None
        mask = None

        if max_num_frames > 0:
            n_frames = max_num_frames
        else:
            n_frames = params['timestamps'].shape[0]

        for i in tqdm.trange(n_frames):

            frame = cv2.cvtColor(reader.get_data(i), cv.CV_RGB2GRAY)

            if i == 0:

                if bbox is None or len(bbox) == 0:

                    bbox, mask = select_ROI_and_mask(frame)
                    if len(bbox) == 0:
                        bbox = [0, 0, frame.shape[1], frame.shape[0]]

                print "bounding box:", bbox

                size = (n_frames, bbox[3], bbox[2])
                fp = np.memmap(mmap_file, dtype='uint8', mode='w+', shape=size)

            if mask is not None:
                frame[~mask] = 255

            if len(bbox) > 0:
                frame = frame[bbox[1]:bbox[1]+bbox[3],
                              bbox[0]:bbox[0]+bbox[2]]

            fp[i, :, :] = frame

        # make sure to flush file
        del fp

        dd = {'file': mmap_file,
              'bbox': bbox,
              'n_frames': n_frames,
              'total_width': params['width'],
              'total_height': params['height'],
              'width': bbox[2],
              'height': bbox[3],
              'w_offset': bbox[0],
              'h_offset': bbox[1],
              'dtype': 'uint8',
              'size': size,
              'timestamps': params['timestamps'][:n_frames]}

        with open(param_file, 'w') as f:
            pickle.dump(dd, f)

    else:

        with open(param_file, 'r') as f:
            dd = pickle.load(f)

    return dd
