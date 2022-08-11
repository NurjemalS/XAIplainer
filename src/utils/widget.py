from dataclasses import dataclass
from typing import Any, List, Optional, Union, Type
import streamlit as st




@dataclass
class InputWidgetOption:
    name: str
    label: str
    value: Optional[Union[List[Any], Any]] = None
    is_text_input: bool = False
    data_type: Optional[Type] = None
    min_value: float = None
    max_value: float = None
    step: float = None

    def render(self):
        
         if isinstance(self.value, list):
            return st.selectbox(
                label=self.name,
                options=self.value,
                key=self.label
            )
         elif self.is_text_input and self.data_type != str:
            return st.number_input(
                label=self.name,
                min_value=self.min_value,
                max_value=self.max_value,
                value=self.value,
                step=self.step,
                key=self.label
            )
         elif self.data_type == str:
             return st.text_input(
                label=self.name,  
                value=self.value,
                key=self.label
            )

         else:
            raise ValueError(f"Cant render {self.label}")
        
