import sys
from app import App


def main():
    app = App(sys.argv)
    app.run()


if __name__ == "__main__":
    main()
