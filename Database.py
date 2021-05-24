import json

def convert_to_list(filepath):
    with open(filepath) as file:
        data = json.load(file)
    
    addresses = []
    for name,street_name,city,state,zipCode,extnZip,country in zip(data['name'],data['streetName'],data['city'],data['state'],data['zipCode'],data['extnZip'],data['country']):
        addresses.append(name+", "+street_name+", "+city+", "+state+", "+zipCode+", "+extnZip+", "+country)
    
    return addresses
    

print(convert_to_list("calhounorders-deliveryAddress.json"))

