from datetime import datetime, timedelta
import pytz

available_shoes = {
    '1': {'name': 'Running Shoes', 'price': 50.0},
    '2': {'name': 'Casual Sneakers', 'price': 40.0},
    '3': {'name': 'Formal Shoes', 'price': 60.0},
}
shopping_cart = {}

def get_current_datetime_ist():
    utc_now = datetime.utcnow()
    ist_timezone = pytz.timezone('Asia/Kolkata')
    ist_now = utc_now + timedelta(seconds=ist_timezone.utcoffset(utc_now).seconds)

    ist_formatted = ist_now.strftime('%Y-%m-%d %H:%M:%S %Z')
    return f'The current date and time in IST is: {ist_formatted}'

def show_available_shoes():
    print("Available Shoes:")
    for shoe_id, shoe_info in available_shoes.items():
        print(f"{shoe_id}. {shoe_info['name']} - ${shoe_info['price']:.2f}")

def calculate_total_cart_value():
    total_value = sum(item['price'] * item['quantity'] for item in shopping_cart.values())
    return f'Total Cart Value: ${total_value:.2f}'

def display_date_and_time_of_purchase():
    return get_current_datetime_ist()

def buy_items():
    while True:
        if not shopping_cart:
            print('Your shopping cart is empty. Add items before purchasing.')
            break

        print("Items in your shopping cart:")
        for item_id, item_info in shopping_cart.items():
            print(f"{item_info['quantity']} x {item_info['name']} - ${item_info['price']:.2f} each")

        total_value = sum(item['price'] * item['quantity'] for item in shopping_cart.values())
        print(f'Total Cart Value: ${total_value:.2f}')

        confirmation = input("Do you want to proceed with the purchase? (yes/no): ").lower()
        if confirmation == 'yes':
            print('Purchase successful!')
            shopping_cart.clear()
            break
        elif confirmation == 'no':
            print('Purchase cancelled.')
            break
        else:
            print('Invalid input. Please enter "yes" or "no".')

def simple_chatbot():
    print("Hi, I am Chatbot! I am here to assist you with Shoe Shopping.")

    while True:
        print("Commands:")
        print("1. Show available shoes")
        print("2. Buy")
        print("3. Total Cart Value")
        print("4. Date and Time Of Purchase")
        print("5. Exit")

        user_input = input('You: ')
        user_input_lower = user_input.lower()

        if user_input_lower in ['exit', 'quit']:
            print('Chatbot: Goodbye!')
            break

        if user_input_lower == '1':
            show_available_shoes()
        elif user_input_lower == '2':
            while True:
                shoe_id = input('Enter the shoe ID to add to cart (or type "done" to finish): ')
                if shoe_id.lower() == 'done':
                    break
                quantity = int(input('Enter the quantity: '))
                if shoe_id in available_shoes:
                    shoe = available_shoes[shoe_id]
                    if shoe_id in shopping_cart:
                        shopping_cart[shoe_id]['quantity'] += quantity
                    else:
                        shopping_cart[shoe_id] = {'name': shoe['name'], 'price': shoe['price'], 'quantity': quantity}
                    print(f'Added {quantity} {shoe["name"]} to the cart.')
                else:
                    print('Invalid shoe ID. Please choose a valid shoe.')
        elif user_input_lower == '3':
            print('Chatbot:', calculate_total_cart_value())
        elif user_input_lower == '4':
            print('Chatbot:', display_date_and_time_of_purchase())
        elif user_input_lower == '5':
            buy_items()
        else:
            print('Chatbot: Invalid command. Please choose a valid command.')

simple_chatbot()
