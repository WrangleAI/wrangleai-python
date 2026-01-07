from typing import Any, Dict, List, Optional, Union

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

WrangleModel = Union[
    Literal["auto", "gpt-4", "gpt-4o", "gpt-4o-mini", "gemini-1.5-pro"], 
    str
]

class SLMConfig(TypedDict, total=False):
    """
    Configuration for Efficiency-First Routing.
    
    Attributes:
        useSlm (bool): Enable routing to Small Language Models.
        useCase (str, optional): The specific domain (e.g., 'coding', 'chat').
    """
    useSlm: bool
    useCase: Optional[str]

class WrangleObject:
    def __init__(self, data: Any):
        self._data = data
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    setattr(self, key, WrangleObject(value))
                else:
                    setattr(self, key, value)

    def __len__(self):
        if isinstance(self._data, (list, dict, str)):
            return len(self._data)
        return 0

    def __getattr__(self, name):
        return None
    
    def __getitem__(self, index):
        if isinstance(self._data, list):
            val = self._data[index]
            return WrangleObject(val) if isinstance(val, (dict, list)) else val
        raise TypeError("WrangleObject is not subscriptable")

    def __iter__(self):
        if isinstance(self._data, list):
            for item in self._data:
                yield WrangleObject(item) if isinstance(item, (dict, list)) else item
        else:
            raise TypeError("WrangleObject is not iterable")

    def __repr__(self):
        return f"{self._data}"
    
    def to_dict(self) -> Any:
        return self._data