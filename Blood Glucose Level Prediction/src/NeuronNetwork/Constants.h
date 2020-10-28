//
//  Constants.h
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 17/10/2020.
//

#ifndef Constants_h
#define Constants_h

namespace Constants
{
    static constexpr double Low_Threshold = 3.0;            //mmol/L below which a medical attention is needed
    static constexpr double High_Threshold = 13.0;            //dtto above
    static constexpr size_t Internal_Bound_Count = 32;      //number of bounds inside the thresholds

    static constexpr double Band_Size = (High_Threshold - Low_Threshold)
                                            / static_cast<double>(Internal_Bound_Count); //must imply relative error <= 10%
    static constexpr double Inv_Band_Size = 1.0 / Band_Size;        //abs(Low_Threshold-Band_Size)/Low_Threshold
    static constexpr double Half_Band_Size = 0.5 / Inv_Band_Size;
    static constexpr size_t Band_Count = Internal_Bound_Count + 2;

    double Band_Index_To_Level(const size_t index)
    {
        if (index == 0) return Low_Threshold - Half_Band_Size;
        if (index >= Band_Count - 1) return High_Threshold + Half_Band_Size;

        return Low_Threshold + static_cast<double>(index - 1) * Band_Size + Half_Band_Size;
    }

    /**
        Mapování konkrétní hodnoty mmol/l na výstup neuronky
     */
    unsigned Level_To_Index_Band(const double value)
    {
        // pokud <= 3 odpovídá index 0
        if (value <= 3.0)
        {
            return 0;
        }
        
        // pokud >= odpovídá index 31 (vektror od indexu 0, celkově 32 -> 32 - 1)
        if (value >= 13.0)
        {
            return 31;
        }
        
        int index = 0;
        double start = 3;
        double tmp = 0;
        
        // asi by šlo líp
        // průběžně přičítám velikost intervalu, pokud zadaná hodnota je větší, přičítám pořád
        // pokud je zadaná hodnota menší, vím že hodnota je z intervalu na pozici index
        while (index < 32)
        {
            if (start > value)
            {
                break;
            }
            
            index++;
            start += Band_Size;;
        }
        
        return index;
    }

}


#endif /* Constants_h */
