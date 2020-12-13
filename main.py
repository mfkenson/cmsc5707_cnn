import librosa
import skimage.io
import pathlib
from fastai.vision.all import *


def scale_minmax(X, min= 0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def get_path_class(dat, filename):
    excerpt = dat[dat['slice_file_name'] == filename]
    path_name = os.path.join('UrbanSound8K/audio', 'fold' + str(excerpt.fold.values[0]), filename)
    return path_name, excerpt['class'].values[0]


def save_spectrogram_image(X, sr, out, hop_length=512, n_mels=256):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=X, sr=sr, n_mels=n_mels, n_fft=hop_length * 2, hop_length=hop_length)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)
    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy
    skimage.io.imsave(out, img)


def extract_feature():
    data = pd.read_csv("UrbanSound8K/metadata/UrbanSound8K.csv")
    for i in range(data.shape[0]):
        full_path, class_id = get_path_class(data, data.slice_file_name[i])
        X, sample_rate = librosa.load(full_path, res_type='kaiser_fast')
        out_path = 'spec_images_all/' + data.slice_file_name[i].replace(".wav", ".png")
        save_spectrogram_image(X, sr=sample_rate, out=out_path)


def interpret_cnn(learner):
    interp = ClassificationInterpretation.from_learner(learner)
    # confusion matrix
    interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)
    # most confused
    interp.most_confused(min_val=5)


def label_func(f):
    return (f.split('-'))[1]


def train_cnn():
    path = pathlib.Path.cwd()
    path = path / 'spec_images_all'
    files = get_image_files(path)
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224), bs=64)
    dls.show_batch()
    learn_all = cnn_learner(dls, alexnet, metrics=error_rate, normalize=True, pretrained=False)
    learn_all.fine_tune(100)
    # save pkl
    learn_all.export(os.path.abspath('./cnn_model.pkl'))
    # interpret
    interpret_cnn(learn_all)


def inference_cnn(file_path):
    label_lookup = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
                    'enginge_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    learner = load_learner('cnn_model.pkl')
    print('Incoming file:', file_path)
    if file_path.lower().endswith('.wav'):
        #encoded in png
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        out_path = 'tmp.png'
        save_spectrogram_image(X, sr=sample_rate, out=out_path)
        file_path = out_path
    class_id, _, _ = learner.predict(file_path)
    print('Predicted Class Label:', label_lookup[int(class_id)])


if __name__ == '__main__':
    # PROGRAM_MODE = 'EXTRACT_FEATURE'
    # PROGRAM_MODE = 'TRAIN_CNN'
    PROGRAM_MODE = 'INFERENCE_CNN'

    #INFERENCE_FILE_PATH = 'spec_images_all/518-4-0-0.png' #taken from original dataset
    INFERENCE_FILE_PATH = 'test_set_wav/labrador-barking-daniel_simon.wav' #from internet
    #INFERENCE_FILE_PATH = 'test_set_wav/gun_battle_sound-ReamProductions-1158375208.wav'#from internet
    if PROGRAM_MODE == 'EXTRACT_FEATURE':
        extract_feature()
    elif PROGRAM_MODE == 'TRAIN_CNN':
        train_cnn()
    elif PROGRAM_MODE == 'INFERENCE_CNN':
        inference_cnn(INFERENCE_FILE_PATH)
