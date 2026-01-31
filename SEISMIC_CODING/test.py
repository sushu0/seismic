# /d:/xiangmu/项目/SEISMIC_CODING/test.py
def main():
    # simple string
    print("Hello, world!")

    # multiple values, custom separator and end
    print("A", "B", "C", sep="-", end=".\n")

    # formatted output (f-string)
    name = "Seismic"
    magnitude = 4.7
    print(f"Event: {name}, Magnitude: {magnitude:.1f}")

    # printing a list on one line
    items = [1, 2, 3]
    print("Items:", *items)

    # multi-line string
    print("""\
Multi-line
text block
example
""")

if __name__ == "__main__":
    main()