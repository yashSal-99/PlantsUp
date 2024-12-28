from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.filemanager import MDFileManager
from kivy.uix.image import Image
from kivy.lang import Builder
from kivymd.uix.label import MDLabel
from model import predict_plant_species

Builder.load_file('design.kv')


class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path,
            preview=True,
        )
        self.selected_image_path = None

    def upload_image(self):
        """Open the file manager to select an image."""
        self.file_manager.show('C:\yash\PlantsUp')  # You can specify the starting path

    def select_path(self, path):
        """Handle the selected file."""
        self.selected_image_path = path
        self.ids.image_display.source = path
        self.exit_manager()

    def exit_manager(self, *args):
        """Close the file manager."""
        self.file_manager.close()

    def process_image(self):
        """Process the selected image."""
        if self.selected_image_path:
            result_text = "Name of PLANT Species is : "+predict_plant_species([str(self.selected_image_path)])
            #result_text = f"Processed image: {self.selected_image_path.split('/')[-1]}"
            self.ids.result_label.text = result_text
        else:
            self.ids.result_label.text = "Please select an image first."


class PlantsUpApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.primary_hue = "400"
        self.theme_cls.accent_palette = "Yellow"
        self.theme_cls.theme_style = "Light"
        return MainScreen()


if __name__ == "__main__":
    PlantsUpApp().run()
