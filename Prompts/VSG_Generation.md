    Analyze the input image and its caption to identify all the objects, their attributes, and the relationships between them.
    
    1. Identify Objects: Label each object with a specific name (e.g., 'red apple' instead of 'fruit', 'fred' or 'wilma' or 'dino' instead of man, woman, animal).
    2. Define Attributes: For each identified object, list attributes such as color, size, and position in the format:
    - Object (attr1, attr2, attr3, ...)
    3. Specify Relationships: Describe relationships between objects in the format:
    - relationship(obj1, obj2)
    4. The characters depicted in the image are restricted to the following list: ['fred', 'wilma', 'barney', 'betty', 'pebble', 'slate', 'dino', 'bamm-bamm'].

    Caption : {sample['scene_description']}

    Return a single object if multiple similar objects are detected.

    Return the result in the following JSON format without any other content:

    {{
        "objects": [
            {{"name": "object1", "attributes": ["attr1", "attr2", ...]}},
            {{"name": "object2", "attributes": ["attr1", "attr2", ...]}}
        ],
        
        "relationships": [
            "relationship(object1, object2)",
            "relationship(object3, object4)"
        ]
    }}
