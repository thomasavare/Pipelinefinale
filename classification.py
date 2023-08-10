from transformers import DistilBertTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from numpy import argmax

id2label = {0: 'ALUMINIUM CAN', 1: 'ALUMINIUM SHEET', 2: 'ALUMINIUM TRAY', 3: 'CIGARETTE BUTT',
            4: 'CIGARETTE PACK', 5: 'COMPOSTABLE PACKAGING', 6: 'CONDIMENT PACKETS', 7: 'COVID TEST',
            8: 'CRUMBLED TISSUE', 9: 'CYLINDRICAL BATTERY', 10: 'FACE MASK', 11: 'GLASS BOTTLE',
            12: 'GLASS JAR', 13: 'LAPTOP CHARGER', 14: 'MEDS BLISTER', 15: 'METAL CAP',
            16: 'MIXED PAPER-PLASTIC PACKAGING', 17: 'ORGANIC SCRAPS', 18: 'PAPER BOWL',
            19: 'PAPER CUP', 20: 'PAPER FOOD PACKAGING', 21: 'PAPER MAGASINE',
            22: 'PAPER PACKAGING', 23: 'PAPER PLATE', 24: 'PAPER SHEET', 25: 'PAPER SUGAR BAG',
            26: 'PAPER TRAY', 27: 'PHONE CHARGER', 28: 'PIZZA BOX', 29: 'PLASTIC BAG',
            30: 'PLASTIC BOTTLE', 31: 'PLASTIC BOWL', 32: 'PLASTIC CAP', 33: 'PLASTIC CUP',
            34: 'PLASTIC CUTLERY', 35: 'PLASTIC DISH', 36: 'PLASTIC GLOVES',
            37: 'PLASTIC PACKAGING', 38: 'PLASTIC SNACK PACKAGING', 39: 'PLASTIC STICKS',
            40: 'PLASTIC STRAW', 41: 'PLASTIC TRAY', 42: 'RECEIPT', 43: 'SMARTPHONE',
            44: 'TEA BAG', 45: 'TETRAPACK', 46: 'TOBACCO PACK', 47: 'TRANSPORT TICKET',
            48: 'WOODEN CUTLERY', 49: 'WOODEN STICKS'}


def load_bert():
    model = TFAutoModelForSequenceClassification.from_pretrained("thomasavare/distilbert-ft-test3")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer, model


def classify(text, tokenizer, model, to_class=False):
    x = [text]
    tokenized = tokenizer(x)
    tfdataset = tf.data.Dataset.from_tensor_slices(dict(tokenized))
    tfdataset = tfdataset.batch(1)
    res = model.predict(tfdataset).logits
    if to_class:
        return id2label[argmax(res)]
    return argmax(tf.nn.softmax(tf.convert_to_tensor(res)).numpy())


if __name__ == "__main__":
    tokenizer, model = load_bert()
    text = input("text to classify: ")
    print(classify(text, tokenizer, model))
