import numpy as np,pandas as pd,matplotlib.pyplot as plt,seaborn as sns,tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from src.config import *
def load_test_data():
    df=pd.read_csv(DATA_FILE); x=np.zeros((len(df),IMG_SIZE,IMG_SIZE,1),dtype=np.float32)
    for i,s in enumerate(df.pixels): x[i,:,:,0]=np.fromstring(s,dtype=np.float32,sep=' ').reshape(IMG_SIZE,IMG_SIZE)/255.0
    y=df.emotion.astype(int).to_numpy(); _,xt,_,yt=train_test_split(x,y,test_size=.15,random_state=42,stratify=y); return xt,yt
def main():
    if not DATA_FILE.exists(): raise FileNotFoundError('data/fer2013.csv not found.')
    if not MODEL_FILE.exists(): raise FileNotFoundError("Model not found. Run 'python train.py' first.")
    x, y = load_test_data()
    m = tf.keras.models.load_model(str(MODEL_FILE))
    pred = np.argmax(m.predict(x, verbose=0), axis=1)
    print(classification_report(y,pred,target_names=CLASS_NAMES,zero_division=0)); matrix=confusion_matrix(y,pred); RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(8,6)); sns.heatmap(matrix,annot=True,fmt='d',xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Emotion Classification Confusion Matrix'); plt.tight_layout(); out=RESULTS_DIR/'confusion_matrix.png'; plt.savefig(out,dpi=150); plt.close(); print('Saved:',out)
if __name__=='__main__': main()
