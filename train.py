import argparse,numpy as np,pandas as pd,tensorflow as tf
from sklearn.model_selection import train_test_split
from src.config import *
from src.utils import save_history,plot_history
def load_data(csv_file,limit=0):
    if not csv_file.exists(): raise FileNotFoundError('Place FER2013 at data/fer2013.csv')
    df=pd.read_csv(csv_file)
    if not {'emotion','pixels'}.issubset(df.columns): raise ValueError('CSV must contain emotion and pixels columns.')
    if limit>0: df=df.sample(n=min(limit,len(df)),random_state=42)
    x=np.zeros((len(df),IMG_SIZE,IMG_SIZE,1),dtype=np.float32)
    for i,s in enumerate(df.pixels):
        values=np.fromstring(s,dtype=np.float32,sep=' ')
        if len(values)!=IMG_SIZE*IMG_SIZE: raise ValueError(f'Invalid pixel count at row {i}.')
        x[i,:,:,0]=values.reshape(IMG_SIZE,IMG_SIZE)/255.0
    return x,df.emotion.astype(int).to_numpy()
def build_model():
    m=tf.keras.Sequential([tf.keras.layers.Input((IMG_SIZE,IMG_SIZE,1)),tf.keras.layers.Conv2D(32,3,activation='relu',padding='same'),tf.keras.layers.BatchNormalization(),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Dropout(.25),tf.keras.layers.Conv2D(64,3,activation='relu',padding='same'),tf.keras.layers.BatchNormalization(),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Dropout(.25),tf.keras.layers.Conv2D(128,3,activation='relu',padding='same'),tf.keras.layers.BatchNormalization(),tf.keras.layers.MaxPooling2D(),tf.keras.layers.Dropout(.3),tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.Dropout(.4),tf.keras.layers.Dense(NUM_CLASSES,activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='sparse_categorical_crossentropy',metrics=['accuracy']); return m
def main():
    p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=12000); p.add_argument('--epochs',type=int,default=EPOCHS); a=p.parse_args()
    np.random.seed(42); tf.random.set_seed(42); x,y=load_data(DATA_FILE,a.limit)
    xtr,xt,ytr,yt=train_test_split(x,y,test_size=.15,random_state=42,stratify=y); xtr,xv,ytr,yv=train_test_split(xtr,ytr,test_size=.15,random_state=42,stratify=ytr)
    print(f'Training: {len(xtr)} | Validation: {len(xv)} | Test: {len(xt)}'); m=build_model(); MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_FILE), monitor='val_accuracy', save_best_only=True)
    ]
    h = m.fit(xtr, ytr, validation_data=(xv, yv), epochs=a.epochs, batch_size=BATCH_SIZE, callbacks=cb, verbose=1)
    loss, acc = m.evaluate(xt, yt, verbose=0)
    print(f'Test Accuracy: {acc * 100:.2f}%')

    # Ensure best model is saved (especially when EarlyStopping restored best weights)
    m.save(str(MODEL_FILE))
    hd = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    save_history(hd, RESULTS_DIR / 'training_history.json')
    plot_history(hd, RESULTS_DIR / 'training_history.png')
    print('Model saved:', MODEL_FILE)

if __name__=='__main__': main()
