import torch, torch.nn as nn, numpy as np, sqlite3
from NNArchitecture import AmazonNeuralNetwork
import joblib 

gpu_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model() -> torch.nn.Module:
    # model = AmazonNeuralNetwork().to(device=gpu_device)
    # model.load_state_dict(torch.load('AmazonNeuralNet.pth', map_location=gpu_device))

    #Use ONLY `.load(...)` if you `torch.save()`'d the entire class, not just weights. 
    model = AmazonNeuralNetwork().to(gpu_device)
    # model = torch.load("AmazonNeuralNet.pth", map_location=gpu_device, weights_only=False)
    model.load_state_dict(
        torch.load(
            "AmazonNeuralNet.pth",
            map_location=gpu_device,
            weights_only=True
        )
    )
    # model.load("AmazonNeuralNet.pth", map_location=gpu_device, weights_only=False)
    return model 


if __name__=='__main__':

    def get_valid_int(prompt: str, lower: int, upper: int) -> int:
        while True:
            try:
                value = int(input(prompt))
            except ValueError:
                print(f'Please enter an integer between {lower} and {upper}.')
                continue

            if lower <= value <= upper:
                return value

            print(f'Value must be between {lower} and {upper}.')

    def get_valid_day_for_month(month: int) -> int:
        days_in_month = {
            1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        while True:
            day = get_valid_int(
                f'Enter a day for month {month}\n(ex: for April 4th, enter 4): ',
                1,
                days_in_month[month]
            )
            return day

    # First 9 input features must stay within realistic ranges before model prediction.
    day_of_week = get_valid_int('Enter a day of the week\n(ex: 1 for Monday, 2 for Tuesday, ..., 7 for Sunday): ', 1, 7)
    day_of_week_encoded = day_of_week - 1
    month = get_valid_int('Enter a number for the month\n(ex: 1 for January, 2 for February, ..., 12 for December): ', 1, 12)
    day = get_valid_day_for_month(month)
    hour = get_valid_int('Enter the hour of day in military time (0-23): ', 0, 23)
    is_weekend = 1 if day_of_week_encoded >= 5 else 0
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    #Extract the prev_order's cluster encode label, then extract the day to calculate `days_since_prev_order`
    sqlQUERY = f"""
    SELECT Cluster_Encoding, day FROM Amazon_DataSet WHERE month = {month} AND day <= {day} AND hour <= {hour};
    """

    conn = sqlite3.connect('local.db')
    cursor = conn.cursor()
    cursor.execute(sqlQUERY)

    #Should ONLY fetch one record.     
    item = cursor.fetchone()
    #Label it -1 in this branch. 
    if item is None:
        print('No item returned from SQL query.')
        prev_cluster = -1 
        days_since_prev_order = 0.0 
    #Label it based on item[0], then subtract user's days from the prev_record_day 
    else:
        prev_cluster = item[0]
        prev_record_day = item[1]
        days_since_prev_order = day - prev_record_day
        print(prev_cluster)
        print(prev_record_day)
    
    #Create 16 one-hot encodings for prev_cluster values from -1 to 14.
    # This must match the training feature order exactly: [-1, 0, 1, ..., 14].
    def dummy_encoding_gen():
        for i in range(-1, 15):
            if i == prev_cluster:
                yield 1
            else:
                yield 0

    dummy_encodings = [zero for zero in dummy_encoding_gen()]
    
    #Create a finalzied list 
    #`prev_cluster` is not used, as it was only needed to generate the dummy encodings.
    user_input_list = [[day_of_week_encoded, month, day, hour, is_weekend, month_sin, month_cos,
     hour_sin, hour_cos, days_since_prev_order] + dummy_encodings]

    #Import Scaler and use for input row.
    scaler = joblib.load("scaler.pkl")
    user_input_list = scaler.transform(user_input_list)

    #Prepare user input row as a torch.tensor object 
    user_input_tensor = torch.tensor(user_input_list, dtype=torch.float32, device=gpu_device) 
    print(user_input_tensor, len(user_input_list))

# TURN ON EVAL() mode for predicting values. ===============================
    model = load_model()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Turn off backward prop. Only forward pass (prediction)
    with torch.no_grad():
        outputs = model(user_input_tensor.cuda())
        #Pass the tensor obj back to the CPU for more Python Programming 
        y_pred = int(torch.argmax(outputs).cpu())
        print('Category Encoding Label:' ,y_pred)

    category_dict = {
        0 : 'Haircare',
        1 : 'Clothing Accessories (Watches, Hats, etc.)',
        2 : 'Shoes or Sandals',
        3 : 'Kitchen Supplies',
        4 : 'Pantry Snacks or Kitchen Goods',
        5 : 'Skincare',
        6 : 'Vitamins or Nutritional Supplements',
        7 : 'SD Cards, Chargers, or Cables',
        8 : 'House Goods',
        9 : 'Video Games',
        10 : 'Shirts',
        11 : 'Styling Products or Makeup',
        12 : 'Electronics Accessories or Printer Ink',
        13 : 'Women\'s Clothing',
        14 : 'Miscallaneous'
    }
    print(category_dict[y_pred])
    
    



         
    

