"""Fixed Point command FI

	This function provides the same solution as ML's fi command.
	The difference is that the output can actually be printed in the
	screen rather than creating an object. This function also have the
	option to return a list of values with the desired type of conversion.
	This function also provide error tracing, showing the values that are
	not corrected input for the configuration set by the user. The first
	value display will be in the input format.
	- This code can be used 2 ways:
	1- As a library by adding the following line in your code:

							from fi import fi

		IMPORTANT: remember to have this file in the same folder as the
					file you will be calling fi function from
	2- As a standalone conversion code. Hence you can run this code using
	command line like shown below:

							python fi.py

		IMPORTANT: remember to have terminal/prompt opened in the same
		folder as this file
	Parameters:
		- Values: The input values of the function.
				  * The input is a PYTHON LIST of any size
				  	(can accept also one INT or FLOAT)
				  * For dec to hex or bin conversion the values need to be in
				   dec format and for hex or bin to dec conversion values
				   need to be bigger than 1
		- Signed: Check if the value is signed (1) or unsigned (0)
		- TotalLen: The total number of bits used
		- FracLen: The number of bits used to represent the fractional part
		- Format: Check if the input is decimal (1) or hex/bin (0)
		- ReturnVal: Check if the function returns value or just prints
					* "None": returns None, only prints values
					* "Dec" : returns a list of decimal values
					* "Hex" : returns a string of hex values
					* "Bin" : returns a string of bin values
	Notes: There are some examples in the bottom of the function. To use them
		   UNCOMMENT the necessary code.
	---
	TODO LIST
		[ ] 1. Return value to not text format
"""
import math
import numpy as np


def fi(Values, Signed, TotalLen, FracLen, Format=1, ReturnVal="None"):
    # Converting values
    dec_text = ""
    hex_text = ""
    bin_text = ""
    dec_vals = []
    if ReturnVal == "None":
        # Printing header
        print("\nFIXED POINT CONVERSION\n")
        if Format:
            print("-Type of conversion:", "Decimal to Hex/Bin")
        else:
            print("-Type of conversion:", "Hex/Bin to Decimal")
        if Signed:
            print("-Signedness:", "Signed")
        else:
            print("-Signedness:", "Unsigned")
        print("-Total bits:", TotalLen)
        print("-Fractional bits:", FracLen)

    # Calculating fractional digits to represent dec
    Precision = math.ceil(FracLen / 3)
    Precision_txt = "{:." + str(Precision) + "f}"

    # Converting input type to list
    if type(Values) == int or type(Values) == float:
        Values = [Values]
    # Decimal
    if Format:
        # Check if it is signed
        if Signed:
            for val in Values:
                # Check if it is positive value
                if val > 0:
                    dec_text = dec_text + Precision_txt.format(val) + ","
                    # Check if value is above the limit
                    if val > (
                        2 ** (TotalLen - FracLen) - (2 ** (TotalLen - FracLen - 1))
                    ):
                        if ReturnVal == "None":
                            print(
                                "\nERROR: Value is too high, range from",
                                (
                                    2 ** (TotalLen - FracLen)
                                    - (2 ** (TotalLen - FracLen - 1))
                                ),
                                "to",
                                -(
                                    2 ** (TotalLen - FracLen)
                                    - (2 ** (TotalLen - FracLen - 1))
                                ),
                                " ( value:",
                                val,
                                " index:",
                                Values.index(val),
                                ")",
                            )
                        return None
                    elif val == (
                        2 ** (TotalLen - FracLen) - (2 ** (TotalLen - FracLen - 1))
                    ):
                        if ReturnVal == "None":
                            print(
                                "WARNING:",
                                (
                                    2 ** (TotalLen - FracLen)
                                    - (2 ** (TotalLen - FracLen - 1))
                                ),
                                "can not be represented,",
                                round(
                                    (
                                        (
                                            2 ** (TotalLen - FracLen)
                                            - (2 ** (TotalLen - FracLen - 1))
                                        )
                                        - 1 / (2 ** TotalLen)
                                    ),
                                    Precision,
                                ),
                                "will be used instead",
                                "( index:",
                                Values.index(val),
                                ")",
                            )
                        val = (
                            2 ** (TotalLen - FracLen) - (2 ** (TotalLen - FracLen - 1))
                        ) - 1 / (2 ** TotalLen)
                        dec_text = ""
                        dec_text = dec_text + Precision_txt.format(val) + ","
                    num = math.ceil(val * (2 ** (TotalLen - (TotalLen - FracLen))))
                    if num >= (2 ** TotalLen) / 2:
                        num = num - 1
                    # Check if value is less than minimal possible
                    if num <= 0:
                        num = 0
                    hex_text = (
                        hex_text + ("0x" + hex(num)[2:].zfill(int(TotalLen / 4))) + ","
                    )
                    bin_text = bin_text + ("0b" + bin(num)[2:].zfill(TotalLen)) + ","
                    if ReturnVal != "None":
                        dec_vals.append(val)
                # If negative
                else:
                    # Check if value is above the limit
                    if (-1) * val > (
                        2 ** (TotalLen - FracLen) - (2 ** (TotalLen - FracLen - 1))
                    ):
                        if ReturnVal == "None":
                            print(
                                "\nERROR: Value is too low, range from",
                                (
                                    2 ** (TotalLen - FracLen)
                                    - (2 ** (TotalLen - FracLen - 1))
                                ),
                                "to",
                                -(
                                    2 ** (TotalLen - FracLen)
                                    - (2 ** (TotalLen - FracLen - 1))
                                ),
                                " ( value:",
                                val,
                                " index:",
                                Values.index(val),
                                ")",
                            )
                        return None
                    num = (
                        (2 ** TotalLen)
                        + (2 ** (TotalLen - FracLen))
                        + int(
                            val * (2 ** (TotalLen - (TotalLen - FracLen)))
                            - (2 ** (TotalLen - FracLen))
                        )
                    )
                    # Check if value is less than minimal possible
                    if num == 2 ** TotalLen:
                        num = 0
                    hex_text = (
                        hex_text + ("0x" + hex(num)[2:].zfill(int(TotalLen / 4))) + ","
                    )
                    bin_text = bin_text + ("0b" + bin(num)[2:].zfill(TotalLen)) + ","
                    dec_text = dec_text + Precision_txt.format(val) + ","
                    if ReturnVal != "None":
                        dec_vals.append(val)

        # If unsigned
        else:
            for val in Values:
                # Check if it is positive value
                if val < 0:
                    if ReturnVal == "None":
                        print(
                            "\nERROR: Negative value ( value:",
                            val,
                            " index:",
                            Values.index(val),
                            ")",
                        )
                    return None
                if val > 2 ** (TotalLen - FracLen):
                    if ReturnVal == "None":
                        print(
                            "\nERROR: Value is too high ( value:",
                            val,
                            " index:",
                            Values.index(val),
                            ")",
                        )
                    return None
                num = math.ceil(val * (2 ** (TotalLen - (TotalLen - FracLen))) - 1)
                hex_text = (
                    hex_text + ("0x" + hex(num)[2:].zfill(int(TotalLen / 4))) + ","
                )
                bin_text = bin_text + ("0b" + bin(num)[2:].zfill(TotalLen)) + ","
                dec_text = dec_text + Precision_txt.format(val) + ","
                if ReturnVal != "None":
                    dec_vals.append(round(val, Precision))

        # Output Values
        if ReturnVal == "None":
            print("\n-Dec Values:", dec_text[:-1])
            print("\n-Hex Values:", hex_text[:-1])
            print("\n-Bin Values:", bin_text[:-1])

        # Returning values
        if ReturnVal == "Dec":
            return dec_vals
        elif ReturnVal == "Hex":
            return hex_text[:-1]
        elif ReturnVal == "Bin":
            return bin_text[:-1]

    # Hex or Bin
    else:
        # Check if it is signed
        if Signed:
            for val in Values:
                if val < (2 ** (TotalLen - 1)):
                    dec_text = (
                        dec_text
                        + Precision_txt.format(
                            val / (2 ** (TotalLen - (TotalLen - FracLen)))
                        )
                        + ","
                    )
                    if ReturnVal != "None":
                        dec_vals.append(
                            round(
                                (val / (2 ** (TotalLen - (TotalLen - FracLen)))),
                                Precision,
                            )
                        )
                else:
                    dec_text = (
                        dec_text
                        + Precision_txt.format(
                            val / (2 ** (TotalLen - (TotalLen - FracLen)))
                            - (2 ** (TotalLen - FracLen))
                        )
                        + ","
                    )
                    if ReturnVal != "None":
                        dec_vals.append(
                            round(
                                (
                                    val / (2 ** (TotalLen - (TotalLen - FracLen)))
                                    - (2 ** (TotalLen - FracLen))
                                ),
                                Precision,
                            )
                        )
                hex_text = (
                    hex_text + ("0x" + hex(val)[2:].zfill(int(TotalLen / 4))) + ","
                )
                bin_text = bin_text + ("0b" + bin(val)[2:].zfill(TotalLen)) + ","
        # If unsigned
        else:
            for val in Values:
                # Check if it is positive value
                if val < 1 and val != 0:
                    if ReturnVal == "None":
                        print(
                            "\nERROR: Wrong input Value, change the conversion type ( value:",
                            val,
                            " index:",
                            Values.index(val),
                            ")",
                        )
                    return None
                dec_text = (
                    dec_text
                    + Precision_txt.format(
                        val / (2 ** (TotalLen - (TotalLen - FracLen)))
                    )
                    + ","
                )
                hex_text = (
                    hex_text + ("0x" + hex(val)[2:].zfill(int(TotalLen / 4))) + ","
                )
                bin_text = bin_text + ("0b" + bin(val)[2:].zfill(TotalLen)) + ","
                if ReturnVal != "None":
                    dec_vals.append(
                        round(
                            (val / (2 ** (TotalLen - (TotalLen - FracLen)))), Precision
                        )
                    )

        # Output Values
        if ReturnVal == "None":
            print("\n-Bin Values:", bin_text[:-1])
            print("\n-Hex Values:", hex_text[:-1])
            print("\n-Dec Values:", dec_text[:-1])

        # Returning values
        if ReturnVal == "Dec":
            return dec_vals
        elif ReturnVal == "Hex":
            return hex_text[:-1]
        elif ReturnVal == "Bin":
            return bin_text[:-1]


if __name__ == "__main__":
    """1st Example
		1 dec input, signed, 64 bit total, 63 bit fractional
	"""
    fi(-0.000000000123453411323, 1, 64, 63)
    input("\nPress ENTER to close...")
    """2nd Example
		128 hex input, signed, 32 bit total, 31 bit fractional, return decimal values
	"""
    input_values = [
        0x7FFFFFFF,
        0x7FB7CEF8,
        0x7B7A082F,
        0x79467C1C,
        0x771CFC11,
        0x74FD5A32,
        0x72E76976,
        0x70DAFDA1,
        0x6ED7EB40,
        0x6CDE07A9,
        0x6AED28F2,
        0x690525F1,
        0x6725D639,
        0x654F1214,
        0x6380B283,
        0x61BA9137,
        0x5FFC8890,
        0x5E46739C,
        0x5C982E10,
        0x5AF19445,
        0x5952833A,
        0x57BAD88C,
        0x562A7275,
        0x54A12FC8,
        0x531EEFF3,
        0x51A392F5,
        0x502EF961,
        0x4EC10457,
        0x4D599589,
        0x4BF88F2D,
        0x4A9DD406,
        0x49494759,
        0x47FACCF0,
        0x46B24917,
        0x456FA094,
        0x4432B8AE,
        0x42FB7724,
        0x41C9C22C,
        0x409D8072,
        0x3F769917,
        0x3E54F3AD,
        0x3D387835,
        0x3C210F1D,
        0x3B0EA13F,
        0x3A0117E0,
        0x38F85CAB,
        0x37F459B1,
        0x36F4F969,
        0x35FA26AA,
        0x3503CCAC,
        0x3411D707,
        0x332431B0,
        0x323AC8F6,
        0x31558983,
        0x3074605A,
        0x2F973AD2,
        0x2EBE069B,
        0x2DE8B1B5,
        0x2D172A74,
        0x2C495F7D,
        0x2B7F3FC2,
        0x2AB8BA85,
        0x29F5BF55,
        0x29363E09,
        0x287A26C5,
        0x27C169F2,
        0x270BF844,
        0x2659C2B2,
        0x25AABA79,
        0x24FED119,
        0x2455F853,
        0x23B0222A,
        0x230D40E3,
        0x226D46FD,
        0x21D02739,
        0x2135D492,
        0x209E4240,
        0x200963B3,
        0x1F772C96,
        0x1EE790CD,
        0x1E5A8472,
        0x1DCFFBD5,
        0x1D47EB7C,
        0x1CC24822,
        0x1C3F06B5,
        0x1BBE1C54,
        0x1B3F7E52,
        0x1AC32232,
        0x1A48FDA6,
        0x19D1068F,
        0x195B32FD,
        0x18E7792E,
        0x1875CF8B,
        0x18062CA9,
        0x17988749,
        0x172CD656,
        0x16C310E3,
        0x165B2E2E,
        0x15F5259B,
        0x1590EEB6,
        0x152E8132,
        0x14CDD4E8,
        0x146EE1D4,
        0x1411A01A,
        0x13B607FF,
        0x135C11EE,
        0x1303B672,
        0x12ACEE39,
        0x1257B212,
        0x1203FAEF,
        0x11B1C1DF,
        0x11610014,
        0x1111AEDB,
        0x10C3C7A3,
        0x107743F8,
        0x102C1D84,
        0x0FE24E0B,
        0x0F99CF71,
        0x0F529BB5,
        0x0F0CACF0,
        0x0EC7FD57,
        0x0E84873A,
        0x0E424501,
        0x0E013130,
        0x0DC14662,
        0x0D827F4C,
        0x0D44D6BA,
        0x0D084791,
    ]
    output_values = fi(input_values, 1, 32, 31, 0, ReturnVal="Dec")
    print(output_values)
    input("\nPress ENTER to close...")
    """3rd Example
		3 bin input, unsigned, 8 bits total, 0 bit fractional
	"""
    input_values = [0b10000000, 0b10101010, 0b11111111]
    fi(input_values, 0, 8, 0, 0)
    input("\nPress ENTER to close...")
