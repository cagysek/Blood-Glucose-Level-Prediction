//
//  OutputGenerator.cpp
//  Blood Glucose Level Prediction
//
//  Created by Jan Čarnogurský on 22.11.2020.
//

#include "Output_generator.hpp"

#define CIRCLE_SIZE 20
#define CIRCLE_OFFSET_Y_START 60
#define CIRCLE_OFFSET_Y 60
#define CIRCLE_OFFSET_X_START 60
#define CIRCLE_OFFSET_X 400
#define MAX_COUNT 32

#define GRAPH_MIN_SCALE 0
#define GRAPH_MAX_SCALE 255

/**
    Vypočte offset o kolik se musí posunout řada po ose Y aby byla zarovnana na střed
 */
int get_offset(int count)
{
    return (MAX_COUNT - count) / 2;
}

/**
    škálování counteru hrany na rozsah 0-255
 */
int scale_value(double max_value, double min_value, double value)
{
    return (int)((value - min_value) * (GRAPH_MAX_SCALE - GRAPH_MIN_SCALE) / (max_value - min_value) + GRAPH_MIN_SCALE);
}

/**
    Metoda pro generování grafu s prehledem všech hodnot
 */
void Output_generator::generate_graph_transmitted_values(const Neuron_network neural_network)
{
    generate_graph(neural_network, true);
}

/**
    Metoda pro generování grafu s prehledem všech hodnot, které ve výsledku měly chybu po 0,15
 */
void Output_generator::generate_graph_transmitted_values_error(const Neuron_network neural_network)
{
    generate_graph(neural_network, false);
}

/**
    Obecná metoda pro generování grafu,
 */
void Output_generator::generate_graph(const Neuron_network neural_network, bool show_all_transmitted_values)
{
    std::ofstream svg_file;
    
    if (show_all_transmitted_values)
    {
        svg_file.open("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/output/neural_net.svg", std::ios::out | std::ios::trunc);
    }
    else
    {
        svg_file.open("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/output/neural_net_2.svg", std::ios::out | std::ios::trunc);
    }
    
    
    
    if (svg_file.is_open())
    {
        svg_file << "<svg width=\"2000\" height=\"3000\" xmlns=\"http://www.w3.org/2000/svg\">\n";
         
        std::vector<Layer> layers = neural_network.get_layers();
        
        
        // vykreslení neuronů
        for (int i = 0; i < layers.size(); i++)
        {
            int offset = 0;
            int neuron_count = layers[i].get_neuron_count() - 1;
            
            // pokud se nejedna o posledni vrstvu vypočtene offset pro posunutí neuronů
            if (i < layers.size() - 1)
            {
                offset = get_offset(layers[i].get_neuron_count() - 1);
            }
        
            
            for (int j = 0; j < neuron_count; j++)
            {
                int start_y_offset = offset * CIRCLE_OFFSET_Y;
                
                int x_pos = CIRCLE_OFFSET_X_START + CIRCLE_OFFSET_X * i;
                int y_pos = start_y_offset + (CIRCLE_OFFSET_Y_START + j * CIRCLE_OFFSET_Y);
                
                std::string id = "circle_" + std::to_string(i) + "_" + std::to_string(j);
                svg_file << "<ellipse ry=\"" << CIRCLE_SIZE << "\" rx=\"" << CIRCLE_SIZE << "\" id=\""<< id <<"\" cy=\""<< y_pos <<"\" cx=\"" << x_pos << "\" stroke-width=\"1.5\" stroke=\"#000\" fill=\"#fff\"/>\n";
            }
        }
        
        
        double min_val = __DBL_MAX__;
        double max_val = __DBL_MIN__;
        
        // zjištění min a max hodnot
        for (int i = 0; i < layers.size() - 1; i++)
        {
            for (int j = 0; j < layers[i].get_neuron_count() -1; j++)
            {
                for (int k = 0; k < layers[i].get_neuron(j).get_weights().size(); k++)
                {
                    Connection connection = layers[i].get_neuron(j).get_weight(k);
                    
                    double val;
                    if (show_all_transmitted_values)
                    {
                        val = connection.transmitted_value_counter;
                    }
                    else
                    {
                        val = connection.transmitted_value_relative_error_counter;
                    }
                    
                    if (val > max_val)
                    {
                        max_val = val;
                    }
                    else if (val < min_val)
                    {
                        min_val = val;
                    }
                }
            }
        }
        
        
        // vykreslení hran
        for (int i = 0; i < layers.size() - 1; i++)
        {
            
            int offset = get_offset(layers[i].get_neuron_count() - 1);
            
            int color_sum = 0;
            int counter = 0;
            
            // projdu hrany a zjistím střední hodnotu barev kvůli konstantě zvýraznění
            // je to takový nice to have kvůli tomu, že v každy vrstvě jsou nějaký hrany silnější než v ostatních vrstvách
            // díky tomuhle se v každé vrstvě naškáluje zvýraznění nejsilnějších hran
            for (int j = 0; j < layers[i].get_neuron_count() - 1; j++)
            {
                for (int k = 0; k < layers[i].get_neuron(j).get_weights().size(); k++)
                {
                    Connection connection = layers[i].get_neuron(j).get_weight(k);
                    
                    if (show_all_transmitted_values)
                    {
                        color_sum += scale_value(max_val, min_val, connection.transmitted_value_counter);
                    }
                    else
                    {
                        color_sum += scale_value(max_val, min_val, connection.transmitted_value_relative_error_counter);
                    }
                    
                    counter++;
                }
            }
            
            // vypočtu hodnotu zvýraznění
            // střední hodnota + nějaký posun, zajímají nás silnější hrany
            int highlight_from = (color_sum / counter) + 1;
            
            // kreslení hran
            for (int j = 0; j < layers[i].get_neuron_count() - 1; j++)
            {
                int connection_count = layers[i].get_neuron(j).get_weights().size();
                int offset_2 = get_offset(connection_count - 1);
                
                for (int k = 0; k < connection_count; k++)
                {
                    Connection connection = layers[i].get_neuron(j).get_weight(k);
                    
                    int start_y_offset = offset * CIRCLE_OFFSET_Y;
                    int start_y_offset_2 = offset_2 * CIRCLE_OFFSET_Y;
                    
                    // beru pozici kruhu + se posunu doprava o polovinu kruhu
                    int x_1 = CIRCLE_OFFSET_X_START + CIRCLE_OFFSET_X * i + CIRCLE_SIZE;
                    int y_1 = start_y_offset + (CIRCLE_OFFSET_Y_START + j * CIRCLE_OFFSET_Y);
                    
                    
                    int x_2 = CIRCLE_OFFSET_X_START + CIRCLE_OFFSET_X * (i + 1) - CIRCLE_SIZE;
                    int y_2 = start_y_offset_2 + (CIRCLE_OFFSET_Y_START + k * CIRCLE_OFFSET_Y);
                    
                    std::string id = "line_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k);
                    
                    int color_scale;
                    std::string rgb_color;
                    
                    // na základě grafu vyberu hodnoty
                    if (show_all_transmitted_values)
                    {
                        color_scale = scale_value(max_val, min_val, connection.transmitted_value_counter);
                        rgb_color = "rgb(0," + std::to_string(color_scale) + ",0)";
                    }
                    else
                    {
                        color_scale = scale_value(max_val, min_val, connection.transmitted_value_relative_error_counter);
                        rgb_color = "rgb(0,0," + std::to_string(color_scale) + ")";
                    }
                    
                    double line_scale = 0.5;
                    
                    if (color_scale > highlight_from)
                    {
                        line_scale = 1.5;
                    }
                    
                    svg_file << "<line id=\"" << id << "\" y2=\"" << y_2 << "\" x2=\"" << x_2 << "\" y1=\"" << y_1 << "\" x1=\"" << x_1 << "\" fill-opacity=\"null\" stroke-opacity=\"null\" stroke-width=\"" << line_scale << "\" stroke=\"" << rgb_color << "\" fill=\"none\" class=\"connection\" data-value=\"" << connection.transmitted_value_counter << "\" />\n";
                    
                }
                
            }
        }
        
        // skript pro ukázání hodnot hran - nefunkční
        svg_file << "<text id=\"tooltip\" x=\"0\" y=\"0\" visibility=\"hidden\">text</text>\n";
        svg_file << "<script type=\"text/javascript\"><![CDATA["
                    "(function() {"
                        "const lines = document.querySelectorAll(\".connection\");"
                        "for (let i = 0; i < lines.length; i++) {"
                            "lines[i].addEventListener(\"mouseover\", function(evt) {"
                            "tooltip = document.getElementById('tooltip');"
                            "tooltip.setAttributeNS(null,\"x\",evt.clientX+10);"
                            "tooltip.setAttributeNS(null,\"y\",evt.clientY+30);"
                            "tooltip.firstChild.data = this.getAttribute('id') + \": \" + this.getAttribute('data-value');"
                            "tooltip.setAttributeNS(null,\"visibility\",\"visible\");"
                            "console.log(this.getAttribute('id') + \": \" + this.getAttribute('data-value'));"
                        "});"

                        "lines[i].addEventListener(\"mouseout\", function(evt) {"
                            "tooltip = document.getElementById('tooltip');"
                            "tooltip.setAttributeNS(null,\"visibility\",\"hidden\");"
                        " });"
                        "}"
                    "})();"
                    "]]></script>";
        
        svg_file << "</svg>\n";
        svg_file.close();
    }
    else std::cout << "Unable to open file";
}

/**
    Metoda pro generování init souboru
 */
void Output_generator::generate_init_file(const Neuron_network neural_network)
{
    std::ofstream output;
    
    output.open("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/output/neural.ini", std::ios::out | std::ios::trunc);
    
    if (output.is_open())
    {
        std::vector<Layer> layers = neural_network.get_layers();
        
        for (int i = 0; i < layers.size() - 1; i++)
        {
            if (i == 0)
            {
                output << "[hidden_layer_1]\n";
            }
            else if (i == 1)
            {
                output << "[hidden_layer_2]\n";
            }
            else if (i == 2)
            {
                output << "[output_layer]\n";
            }
            
            for (int j = 0; j < layers[i].get_neuron_count(); j++)
            {
                for (int k = 0; k < layers[i].get_neuron(j).get_weights().size(); k++)
                {
                    Connection connection = layers[i].get_neuron(j).get_weight(k);
                    
                    if (k == layers[i].get_neuron(j).get_weights().size() - 1)
                    {
                        output << "Neuron" << j << "_Bias" << "=" << connection.weight << "\n";
                    }
                    else
                    {
                        output << "Neuron" << j << "_Weight" << k << "=" << connection.weight << "\n";
                    }
                    
                }
            }
        }
         
        output.close();
    }
    else
    {
        std::cout << "Nepovedlo se otevřít soubor neural.ini";
    }
    
}

/**
    Metoda pro generování csv souboru s errory
 */
void Output_generator::generate_error_csv(Neuron_network neural_network)
{
    std::ofstream output;
    
    output.open("/Users/cagy/Documents/Škola/PPR/Blood-Glucose-Level-Prediction/Blood Glucose Level Prediction/output/error.csv", std::ios::out | std::ios::trunc);
    
    
    std::vector<double> errors = neural_network.get_errors();
    std::sort(errors.begin(), errors.end());
    
    // vypočtu 1%
    int step = (int)(errors.size() / 100);
    int index = 0;
    
    if (output.is_open())
    {
        output << "Prumerna relativni chyba;" << neural_network.get_average_error() << ";\n";
        output << "standartni odchylka;" << neural_network.get_stanadrd_deviation() << ";\n";
        
        // postupně vypisuju errory po 1%
        while (index < errors.size())
        {
            
            output << errors[index] << ";";
            
            index += step;
        }
        
    }
    else
    {
        std::cout << "Nepovedlo se otevřít soubor pro csv data";
    }
     
}
