from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int


new_person: Person = {"name": "John", "age": 30}
print(new_person)




class Another:
    name: str
    age: int

second: Another = {"name": "John", "age": 30}
print(second)